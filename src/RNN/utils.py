import cPickle
import numpy as np
import theano
import random
from datetime import datetime


IMDB_DATA_DIR = "../../data/imdb"
ROTTOM_DATA_DIR = "../../data/rot_tom"
IMDB_Y = "sentiment"
ROTTOM_Y = "Sentiment"
IMDB_ID = "id"
ROTTOM_ID = "PhraseId"


def reviewWORDS_to_reviewIDX(review, word2idx):
    return [word2idx[w] for w in review]


def reviewIDX_to_reviewWORDS(review, idx2word):
    return [idx2word[idx] for idx in review]


def prepare_data(reviews, labels, maxlen=-1):
    """
    Create the matrices X, X_mask, and Y from set of reviews.
    This pad each review to the same length: the length of the longuest sequence or maxlen.
    :param reviews: list of reviews. each review is a list of numbers (representing words).
    :param labels: list of sentiments for each review.
    :param maxlen: max review length - if <0 no restriction.
    :return: matrices X, X_mask, and Y
    """
    lengths = [len(r) for r in reviews]

    # Labels may be null when passed testing data! (only reviews available)
    if len(labels) > 0:
        assert len(lengths) == len(reviews) == len(labels)
    else:
        assert len(lengths) == len(reviews)

    if maxlen > 0:
        new_lengths = []
        new_reviews = []
        new_labels = []
        if len(labels) > 0:
            for l, r, y in zip(lengths, reviews, labels):
                if l < maxlen:
                    new_lengths.append(l)
                    new_reviews.append(r)
                    new_labels.append(y)
        else:
            for l, r in zip(lengths, reviews):
                if l < maxlen:
                    new_lengths.append(l)
                    new_reviews.append(r)
        lengths = new_lengths
        reviews = new_reviews
        labels = new_labels

    n_samples = len(reviews)
    if n_samples < 1:
        return None, None, None

    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples)).astype('int64')  # this swap the axis!!
    # mask is a matrix of same shape as x, with binary values to know where a review is.
    x_mask = np.zeros((maxlen, n_samples)).astype(theano.config.floatX)

    for i, r in enumerate(reviews):
        x[:lengths[i], i] = r
        x_mask[:lengths[i], i] = 1.

    return x, x_mask, np.asarray(labels)


def make_equal_labels(train_set, n):
    """
    Split the training set into validation set with the requirement that
        there is the same amount (n) of training reviews for each possible label.
    :param train_set: training reviews with their labels.
    :param n: number of training reviews to have for each label.
    :return: training set (with n*num_labels instances) and validation set (with the rest of of the training set).
    """
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = [], []
    new_train_set_x, new_train_set_y = [], []

    num_labels = max(train_set_y) + 1
    num_reviews = np.zeros(num_labels)  # keep track of number of examples for each label.

    # Shuffle TRAIN_X and TRAIN_Y to add randomly selected items to train & validation set.
    seed = datetime.now()
    random.seed(seed)
    random.shuffle(new_train_set_x)
    random.seed(seed)
    random.shuffle(new_train_set_y)

    for x, y in zip(train_set_x, train_set_y):
        if num_reviews[y] >= n:
            # Reached our quota for this label: add to validation set
            valid_set_x.append(x)
            valid_set_y.append(y)
        else:
            # Add to new training set
            new_train_set_x.append(x)
            new_train_set_y.append(y)
            num_reviews[y] += 1.

    assert max(num_reviews) == min(num_reviews) == n

    new_train_set = (np.asarray(new_train_set_x), np.asarray(new_train_set_y))
    valid_set = (np.asarray(valid_set_x), np.asarray(valid_set_y))
    return new_train_set, valid_set


def load_data(dataset, extend_data, n_words=-1, maxlen=-1, valid_portion=0.1, equal_labels=False, sort_by_len=True):
    """
    Load the data and return it in the form of (train, valid, test) sets where each set is of the form (x, y)
    :param dataset: dataset to load - either 'imdb' or 'rottom'.
    :param extend_data: load the extended version of the data (imdb + rottom).
    :param n_words: number of words to keep in the vocabulary - if <0 load all words. All extra words are set to unknow (0).
    :param maxlen: max review length to use in the train/valid set - if <0 no restriction.
    :param valid_portion: proportion of the full train set used for the validation set.
    :param equal_labels: have the same amount of training reviews for each possible label.
    :param sort_by_len: sort train, valid, test sets by review length.
        This allows faster execution as it causes less padding per minibatch.
        Another mechanism must be used to shuffle the train set at each epoch.
    :return: test tsv file, train, valid and test sets along with word2idx and idx2word dictionaries.
    """
    assert dataset in ['imdb', 'rottom']
    print "\nLoading dataset..."
    remove_stop = False
    if dataset == 'imdb':
        data_dir = IMDB_DATA_DIR
        y_index = IMDB_Y
        data_id = IMDB_ID
    elif dataset == 'rottom':
        data_dir = ROTTOM_DATA_DIR
        y_index = ROTTOM_Y
        data_id = ROTTOM_ID
    else:
        print "ERROR: Unknown dataset %s" % dataset
        return

    ###
    # LOAD DICTIONARIES FROM WORDS TO IDX
    ###
    dicts_file_name = data_dir + "/processed/dicts_extended%s_remove-stop%s.pkl" % (
        ("1" if extend_data else "0"),
        ("1" if remove_stop else "0"),
    )
    dicts = cPickle.load(open(dicts_file_name, 'rb'))
    word2idx = dicts[0]
    idx2word = dicts[1]

    ###
    # LOAD IDX DATA
    ###
    data_file_name = data_dir + "/processed/ndata_extended%s_remove-stop%s.pkl" % (
        ("1" if extend_data else "0"),
        ("1" if remove_stop else "0"),
    )
    data = cPickle.load(open(data_file_name, 'rb'))
    train, test = data[0], data[2]  # no use of unlabeled training set now.
    train_reviews = data[3]  # type = list of `reviews`( = list of numbers)
    test_reviews = data[5]  # type = list of `reviews`( = list of numbers)
    print "done."
    print "train:", train.shape, "with", train.columns.values
    print "test:", test.shape, "with", test.columns.values

    # print train_reviews[123]
    # print reviewIDX_to_reviewWORDS(train_reviews[123], idx2word)
    # print ""
    # print test_reviews[123]
    # print reviewIDX_to_reviewWORDS(test_reviews[123], idx2word)

    ###
    # REMOVE EMPTY REVIEWS
    ###
    print "removing empty reviews..."
    train_x = []
    train_y = []
    for x, y in zip(train_reviews, train[y_index]):
        if len(x) > 0 :
            train_x.append(x)
            train_y.append(y)
    train_set = (train_x, train_y)
    print "removed %d empty training reviews" % (len(train_reviews) - len(train_x))

    # DO NOT REMOVE ANY TEST SAMPLES: THEY MUST ALL BE PREDICTED!
    test_set = (np.asarray(test_reviews), None)  # No Y labels for test set

    ###
    # LIMIT REVIEW LENGTH ?
    ###
    if maxlen > 0:
        print "limitting reviews to length %d..." % maxlen
        new_train_x = []
        new_train_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_x.append(x)
                new_train_y.append(y)
        train_set = (new_train_x, new_train_y)
        del new_train_x, new_train_y

    ###
    # LIMIT VOCABULARY SIZE ?
    ###
    if n_words > 0:
        print "limitting vocab to %d words..." % n_words
        train_set_x, train_set_y = train_set
        # test_set_x, test_set_y = test_set

        def limit_vocab(reviews):
            """return set of reviews for which the number of unique words is max `n_words`"""
            return [[0 if w >= n_words else w for w in review] for review in reviews]
        train_set_x = limit_vocab(train_set_x)
        # test_set_x = limit_vocab(test_set_x)
        train_set = (train_set_x, train_set_y)
        # test_set = (np.asarray(test_set_x), test_set_y)

    ###
    # SPLIT TRAIN SET INTO VALIDATION SET
    ###
    print "splitting train set into validation set..."
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    n_train = int(np.round(n_samples * (1. - valid_portion)))

    if equal_labels:
        num_labels = max(train_set[1]) + 1
        num_reviews = np.zeros(num_labels)  # number of examples for each label
        for label in train_set[1]:
            num_reviews[label] += 1.
        print "reviews per label:", num_reviews

        min_reviews = min(num_reviews)  # max amount of training examples for each label.
        print "n_train (%d) / num_labels(%d) = %d" %(n_train, num_labels, n_train/num_labels)
        print "min_reviews", min_reviews
        n_train_per_label = min(n_train/num_labels, min_reviews)
        train_set, valid_set = make_equal_labels(train_set, n_train_per_label)
    else:
        rnd_review_idx = np.random.permutation(n_samples)

        valid_set_x = [train_set_x[i] for i in rnd_review_idx[n_train:]]
        valid_set_y = [train_set_y[i] for i in rnd_review_idx[n_train:]]
        valid_set = (np.asarray(valid_set_x), np.asarray(valid_set_y))

        train_set_x = [train_set_x[i] for i in rnd_review_idx[:n_train]]
        train_set_y = [train_set_y[i] for i in rnd_review_idx[:n_train]]
        train_set = (np.asarray(train_set_x), np.asarray(train_set_y))
    print "train:%s, valid:%s, test:%s" % (train_set[0].shape, valid_set[0].shape, test_set[0].shape)

    ###
    # SORT REVIEWS BY LENGTH
    ###
    if sort_by_len:
        print "sorting reviews by length..."
        train_set_x, train_set_y = train_set
        valid_set_x, valid_set_y = valid_set
        # test_set_x, test_set_y = test_set

        def length_sort_index(reviews):
            """return list of indices for which the reviews' length is increasing"""
            return sorted(range(len(reviews)), key=lambda i: len(reviews[i]))
        sorted_idx = length_sort_index(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_idx]
        train_set_y = [train_set_y[i] for i in sorted_idx]
        train_set = (np.asarray(train_set_x), np.asarray(train_set_y))

        sorted_idx = length_sort_index(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_idx]
        valid_set_y = [valid_set_y[i] for i in sorted_idx]
        valid_set = (np.asarray(valid_set_x), np.asarray(valid_set_y))

        # DON'T CHANGE TEST SET ORDER!! WILL BREAK PREDICTIONS!!
        # sorted_idx = length_sort_index(test_set_x)
        # test_set_x = [test_set_x[i] for i in sorted_idx]
        # test_set_y is NONE! we don't have labels for test set (try on Kaggle).
        # test_set = (np.asarray(test_set_x), test_set_y)

    print "done."
    data = (train_set, valid_set, test_set)
    return test, data, word2idx, idx2word


if __name__ == '__main__':
    train, valid, test, _, _ = load_data('rottom', False, equal_labels=True)
    x, x_mask, y = prepare_data(train[0], train[1])
    print x
    print x.shape
    print x_mask
    print x_mask.shape
    print y
    print y.shape
