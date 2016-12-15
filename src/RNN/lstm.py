from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import argparse
import sys
import os
import time
import lasagne

import numpy as np
import pandas as pd

import theano
from theano import config
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import load_data, prepare_data, reviewIDX_to_reviewWORDS


MODELS_DIR = './LSTM-Models'
PREDICTIONS_DIR = './ModelResponses'


def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Make mini batches of size minibatch_size.
    Used to shuffle the dataset at each iteration.
    :param n: total number of samples
    :param minibatch_size: size of a mini batch
    :param shuffle: shuffle the samples.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return minibatches


def zipp(params, tparams):
    """
    When we reload the model: set param values to tparams
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    Get tparams values into a new params OrderedDict
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = T.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
        state_before * 0.5
    )
    return proj


def init_params(args):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()

    # embedding
    params['Wemb'] = np.random.uniform(-0.01, 0.01, (args.n_words, args.feature_size)).astype(config.floatX)

    # lstm
    params = init_lstm_params(args.feature_size, params)

    # classifier
    params['U'] = np.random.uniform(-0.01, 0.01, (args.feature_size, args.ydim)).astype(config.floatX)
    params['b'] = np.zeros((args.ydim,)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = np.load(path)

    for pf in pp.files:
        print('pp[%s]:' % pf, pp[pf])
    print('pp files:', pp.files)

    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('\n%s is not in the archive %s' % (kk, pp))
        params[kk] = pp[kk]

    return pp


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk, borrow=True)
    return tparams


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def init_lstm_params(feature_size, params):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = np.concatenate([ortho_weight(feature_size),
                        ortho_weight(feature_size),
                        ortho_weight(feature_size),
                        ortho_weight(feature_size)], axis=1)
    params['lstm_W'] = W

    U = np.concatenate([ortho_weight(feature_size),
                        ortho_weight(feature_size),
                        ortho_weight(feature_size),
                        ortho_weight(feature_size)], axis=1)
    params['lstm_U'] = U

    b = np.zeros((4 * feature_size,))
    params['lstm_b'] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, feature_size, mask):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, tparams['lstm_U'])
        preact += x_

        i = T.nnet.sigmoid(_slice(preact, 0, feature_size))
        f = T.nnet.sigmoid(_slice(preact, 1, feature_size))
        o = T.nnet.sigmoid(_slice(preact, 2, feature_size))
        c = T.tanh(_slice(preact, 3, feature_size))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (T.dot(state_below, tparams['lstm_W']) + tparams['lstm_b'])

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[
                                    T.alloc(numpy_floatX(0.), n_samples, feature_size),
                                    T.alloc(numpy_floatX(0.), n_samples, feature_size)
                                ],
                                name='lstm_layers',
                                n_steps=nsteps)
    return rval[0]


def build_model(tparams, args):
    """
    Build prediction and cost functions based on Recurrent LSTM.
    :param tparams: theano.shared parameters to be otimized.
    :param args: options passed to the script.
    :return: x, mask, y matrices, prediction and cost functions.
    """
    trng = RandomStreams(args.seed)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.), borrow=True)

    x = T.matrix('x', dtype='int64')
    mask = T.matrix('mask', dtype=config.floatX)
    y = T.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                args.feature_size])
    proj = lstm_layer(tparams, emb, args.feature_size, mask)
    proj = (proj * mask[:, :, None]).sum(axis=0)
    proj = proj / mask.sum(axis=0)[:, None]
    if args.use_dropout:
        proj = dropout_layer(proj, use_noise, trng)

    prob = T.nnet.softmax(T.dot(proj, tparams['U']) + tparams['b'])
    pred = T.argmax(prob, axis=1)

    f_prob = theano.function([x, mask], prob, name='f_prob', allow_input_downcast=True)
    f_pred = theano.function([x, mask], pred, name='f_pred', allow_input_downcast=True)

    cost = -T.log(prob[T.arange(n_samples), y] + 1e-8).mean()

    return use_noise, x, mask, y, f_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """
    If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    :param f_pred_prob: Theano fct computing the prediction probabilities
    :param prepare_data: usual prepare_data for that dataset.
    :param data: train, or valid tuple (x, y)
    :param iterator: batch of indices
    :param verbose: print status update
    """
    n_samples = len(data[0])
    probs = np.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for indices in iterator:
        reviews = [data[0][t] for t in indices]
        # seqlen = [len(r) for r in reviews]
        labels = np.array(data[1])[indices]  # target labels
        x, mask, _ = prepare_data(reviews, labels)  # get x and mask
        pred_probs = f_pred_prob(x, mask)  # get prediction probabilities
        probs[indices, :] = pred_probs

        n_done += len(indices)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Compute the error
    :param f_pred: Theano fct computing the prediction
    :param prepare_data: usual prepare_data for that dataset.
    :param data: train, or valid tuple (x, y)
    :param iterator: batch of indices
    :param verbose: print status update
    """
    n_samples = len(data[0])
    n_done = 0
    error = 0
    for indices in iterator:
        reviews = [data[0][t] for t in indices]
        # seqlen = [len(r) for r in reviews]
        labels = np.array(data[1])[indices]  # target labels
        x, mask, _ = prepare_data(reviews, labels)  # get x and mask
        preds = f_pred(x, mask)  # get predictions
        error += (preds == labels).sum()

        n_done += len(indices)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    error = 1. - numpy_floatX(error) / len(data[0])

    return error


def pred_to_file(f_pred, prepare_data, data, tsv_file, iterator, file_name, verbose=False):
    """Compute the error
    :param f_pred: Theano function computing the prediction
    :param prepare_data: usual prepare_data for that dataset
    :param data: test data.
    :param iterator: batch of indices
    :param file_name: name of the file to save predictions
    :param verbose: print status update
    """
    n_samples = len(data[0])
    n_done = 0
    preds = []
    for indices in iterator:
        reviews = [data[0][t] for t in indices]
        x, mask, _ = prepare_data(reviews, [])  # get x and mask, no labels available for test data.
        preds.extend(f_pred(x, mask))  # get predictions

        n_done += len(indices)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    print("Writting predictions to file...")
    if 'imdb' in file_name:
        output = pd.DataFrame(data={"id": tsv_file['id'], "sentiment": preds})
    elif 'rottom' in file_name:
        output = pd.DataFrame(data={"PhraseId": tsv_file['PhraseId'], "Sentiment": preds})
    else:
        print("ERROR: Unknown dataset %s" % file_name)
        return

    file_name = file_name.replace(MODELS_DIR, PREDICTIONS_DIR)
    output.to_csv(file_name+'_predictions.csv', index=False, quoting=3)
    print('done.')


def train_lstm():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() in ("yes", "true", "t", "1"))
    parser.add_argument('-v', '--verbose', action="store_true", help="increase output verbosity")
    parser.add_argument('dataset', choices=('imdb', 'rottom'), help='Dataset to load and predict on.')
    parser.add_argument('--extend_data', type='bool', default=False, help='Load the extended version of the data.')
    parser.add_argument('--optimizer', type=str, default='adam', choices=('adam', 'adadelta', 'rmsprop'), help='Optimizer Algo.')
    parser.add_argument('--feature_size', type=int, default=300, help='Number of hidden units.')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of examples to batch in training.')
    parser.add_argument('--valid_batch_size', type=int, default=100, help='Number of examples to batch in validation.')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='Part of training set to use as validation.')
    parser.add_argument('--equal_labels', type='bool', default=False, help='Same amount of training example per label.')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of epochs to run.')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait before early stop.')
    parser.add_argument('--valid_freq', type=int, default=-1, help='Get validation error after this many updates - if <0 compute after each epoch.')
    parser.add_argument('--decay_c', type=float, default=0., help='Decay on the U weights.')
    parser.add_argument('--n_words', type=int, default=-1, help='Vocabulary size - if <0 no restriction.')
    parser.add_argument('--maxlen', type=int, default=-1, help='Max review length - if <0 no restriction.')
    parser.add_argument('--n_recurrent_layers', type=int, default=1, help='Number of recurrent layers.')
    parser.add_argument('--model_fname', type=str, default='lstm-model', help='File name to save / load the model.')
    parser.add_argument('--use_dropout', type='bool', default=True, help='Use drop out.')
    parser.add_argument('--test', type='bool', default=True, help='Load a model (or create one if not present) and make predictions.')
    parser.add_argument('--seed', type=int, default=354592017, help='Random seed.')
    args = parser.parse_args()
    print("\n~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~\nargs:", args)

    np.random.seed(args.seed)

    model_fname = MODELS_DIR + "/" + args.model_fname + "_%s_opt-%s_feature%d_dropout%s_eq-labels%d" % (
        args.dataset + ('-extended' if args.extend_data else ''),
        args.optimizer,
        args.feature_size,
        (1 if args.use_dropout else 0),
        (1 if args.equal_labels else 0)
    )
    args.model_fname = model_fname

    ###
    # LOADING DATASET
    ###
    test_tsv, (train, valid, test), word2idx, idx2word = load_data(
        args.dataset, args.extend_data, args.n_words, args.maxlen, args.valid_portion, args.equal_labels
    )

    # get the number of different classes
    ydim = np.max(train[1]) + 1
    args.ydim = ydim

    # set n_words to the total vocab size
    if args.n_words <= 0:
        args.n_words = len(idx2word)

    # set maxlen to the maximum train review length
    # if args.maxlen <= 0:
    #     lengths = [len(r) for r in train[0]]
    #     args.maxlen = max(lengths)

    ###
    # CREATE / LOAD PARAMETERS
    ###
    print('\nInitializing params and tparams...')
    # Create the initial parameters as numpy ndarrays.
    params = init_params(args)  # Dict name (string) -> numpy ndarray

    existing_model = None  # pre-trained model params, history errors, train & valid errors...
    if args.test and os.path.isfile('%s.npz' % model_fname):
        print('Loading saved params...')
        pp = load_params('%s.npz' % model_fname, params)  # already fine-tuned params
        existing_model = pp

    # Create Theano Shared Variables based on params.
    tparams = init_tparams(params)  # Dict name (string) -> Theano Tensor Shared Variable

    # print("args:", args)
    # print("params:", params)  # Wemb, U, b, lstm_W, lstm_U, lstm_b numpy ndarrays
    # print("tparams:", tparams)  # Wemb, U, b, lstm_W, lstm_U, lstm_b theano shared variables
    total_params = sum([p.get_value().size for p in tparams.values()])
    print("total_params: ", total_params)

    ###
    # BUILD LSTM MODEL
    ###
    print('\nBuilding model...')
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = build_model(tparams, args)

    ###
    # COMPILING THEANO FUNCTIONS
    ###
    print('\nCompiling theano functions...')
    if args.decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(args.decay_c), name='decay_c', borrow=True)
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    if args.optimizer == 'adadelta':
        updates = lasagne.updates.adadelta(cost, tparams.values())
    elif args.optimizer == 'rmsprop':
        updates = lasagne.updates.rmsprop(cost, tparams.values())
    elif args.optimizer == 'adam':
        updates = lasagne.updates.adam(cost, tparams.values())
    else:
        print("unsuported optimizer:", args.optimizer)
        return
    f_train = theano.function([x, mask, y], cost, updates=updates, name='f_train', allow_input_downcast=True)

    ###
    # MAKING PREDICTIONS
    ###
    if existing_model:
        print('Computing training & validation errors...')
        train_batch_idx = get_minibatches_idx(len(train[0]), args.batch_size)
        valid_batch_idx = get_minibatches_idx(len(valid[0]), args.valid_batch_size)
        train_err = pred_error(f_pred, prepare_data, train, train_batch_idx, args.verbose)
        valid_err = pred_error(f_pred, prepare_data, valid, valid_batch_idx, args.verbose)
        print('Train error:', train_err, 'Valid error:', valid_err)

        print('Making predictions on test set...')
        test_batch_idx = get_minibatches_idx(len(test[0]), args.valid_batch_size)
        pred_to_file(f_pred, prepare_data, test, test_tsv, test_batch_idx, model_fname, args.verbose)

    ###
    # TRAINING THE MODEL
    ###
    else:
        print('\nOptimization...')
        valid_batch_idx = get_minibatches_idx(len(valid[0]), args.valid_batch_size)

        # print("%d train examples" % len(train[0]))
        # print("%d valid examples" % len(valid[0]))
        # print("%d test examples" % len(test[0]))

        history_errs = []  # keep track of validation error
        best_p = None  # keep track of best params (numpy ndarrays)
        bad_counter = 0  # keep track of the number of times no improvement was made

        if args.valid_freq < 0:
            args.valid_freq = len(train[0]) // args.batch_size  # test validation accuracy after each batch.
        # print('valid_freq:', args.valid_freq)

        uidx = 0  # the number of updates done
        estop = False  # early stop flag
        start_time = time.time()
        try:
            for eidx in range(args.max_epochs):
                n_samples = 0

                # Get new shuffled index for the training set.
                train_batch_idx = get_minibatches_idx(len(train[0]), args.batch_size, shuffle=False)

                for train_index in train_batch_idx:
                    uidx += 1
                    use_noise.set_value(1.)

                    # Select the random examples for this minibatch
                    x = [train[0][t] for t in train_index]
                    y = [train[1][t] for t in train_index]

                    # Get the data in numpy.ndarray format
                    # This swap the axis! shape = (minibatch maxlen, n samples)
                    x, mask, y = prepare_data(x, y)
                    n_samples += x.shape[1]

                    cost = f_train(x, mask, y)

                    if np.isnan(cost) or np.isinf(cost):
                        print('bad cost detected: ', cost)
                        return 1., 1.

                    if args.verbose:
                        print('Epoch:', eidx, 'Update:', uidx, 'Cost:', cost)

                    # COMPUTE TRAIN & VALIDATION ERRORS
                    if np.mod(uidx, args.valid_freq) == 0:
                        print('Epoch:', eidx, 'Update:', uidx, 'Cost:', cost)
                        print('Computing train and validation errors...')
                        use_noise.set_value(0.)
                        train_err = pred_error(f_pred, prepare_data, train, train_batch_idx, args.verbose)
                        valid_err = pred_error(f_pred, prepare_data, valid, valid_batch_idx, args.verbose)
                        history_errs.append(valid_err)
                        print('Train error:', train_err, 'Valid error:', valid_err, 'history:', history_errs)

                        # VALIDATION SCORE IS IMPROVED
                        if best_p is None or valid_err < min(history_errs[:-1]):
                            best_p = unzip(tparams)
                            bad_counter = 0  # reset counter since we improved!

                            print('Saving...')
                            params = best_p
                            np.savez('%s.npz' % model_fname, history_errs=history_errs, **params)
                            print('Done')

                        # VALIDATION SCORE NOT IMPROVED
                        else:
                            bad_counter += 1
                            if bad_counter > args.patience:
                                print('Early Stop!')
                                estop = True
                                break

                print('Seen %d samples' % n_samples)

                if estop:
                    break

        except KeyboardInterrupt:
            print("Training interupted")

        end_time = time.time()
        if best_p is not None:
            zipp(best_p, tparams)
        else:
            best_p = unzip(tparams)

        use_noise.set_value(0.)
        train_batch_sorted_idx = get_minibatches_idx(len(train[0]), args.batch_size)
        train_err = pred_error(f_pred, prepare_data, train, train_batch_sorted_idx, args.verbose)
        valid_err = pred_error(f_pred, prepare_data, valid, valid_batch_idx, args.verbose)

        print('Train error:', train_err, 'Valid error:', valid_err)
        np.savez(model_fname, train_err=train_err, valid_err=valid_err, history_errs=history_errs, **best_p)
        print('The code ran for %d epochs, with %f sec/epochs' % (
            eidx + 1,
            (end_time - start_time) / (1. * (eidx + 1))
        ))
        print(('Training took %.1fs' % (end_time - start_time)), file=sys.stderr)

        ###
        # MAKING PREDICTIONS ONCE TRAINED
        ###
        if args.test:
            print('Making predictions on test set...')
            test_batch_idx = get_minibatches_idx(len(test[0]), args.valid_batch_size)
            pred_to_file(f_pred, prepare_data, test, test_tsv, test_batch_idx, model_fname, args.verbose)

        return train_err, valid_err


if __name__ == '__main__':
    train_lstm()
