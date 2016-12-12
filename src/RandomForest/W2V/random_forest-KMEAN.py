import argparse
import os
import logging
import pandas as pd
import cPickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from gensim.models import word2vec


IMDB_DATA_DIR = "../../../data/imdb"
ROTTOM_DATA_DIR = "../../../data/rot_tom"


def create_w2v_model(dataset, extend_data, embedding_size, context_size, min_word_count):
    """
    Create a new W2V model and save it for future usage.
    :param dataset: string to decide which dataset to load (imdb or rot-tom).
    :param extend_data: flag to decide to load the extended data (imdb + rot-tom)
    :param embedding_size: int size of word embedding vector (number of features).
    :param context_size: Context / window size.
    :param min_word_count: minimum word count.
    return the new W2V model.
    """
    assert dataset in ['imdb', 'rottom']
    print "\nLoading dataset..."
    remove_stop = False
    extract_tokens = True
    if dataset == 'imdb':
        data_dir = IMDB_DATA_DIR
    elif dataset == 'rottom':
        data_dir = ROTTOM_DATA_DIR
    else:
        print "ERROR: Unknown dataset %s" % dataset
        return

    data_file_name = data_dir + "/processed/data_extended%s_remove-stop%s_tokens%s.pkl" % (
        ("1" if extend_data else "0"),
        ("1" if remove_stop else "0"),
        ("1" if extract_tokens else "0")
    )
    data = cPickle.load(open(data_file_name, 'rb'))

    labeled_train, unlabeled_train, test = data[0], data[1], data[2]
    labeled_train_reviews = data[3]  # type = list of `reviews`( = list of `sentences`( = list of word tokens) )
    unlabeled_train_reviews = data[4]  # type = list of `reviews`( = list of `sentences`( = list of word tokens) )
    test_reviews = data[5]  # type = list of `reviews`( = list of `sentences`( = list of word tokens) )

    sentences = []  # store all sentences from all train reviews
    for review in labeled_train_reviews:
        sentences += review
    if unlabeled_train_reviews:  # None in the case of rot-tom dataset.
        for review in unlabeled_train_reviews:
            sentences += review
    print "done."
    print "number of train sentences %d" % len(sentences)

    print "\nCreating word embeddings..."
    # configure logging so that Word2Vec creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = word2vec.Word2Vec(
        sentences=sentences,
        size=embedding_size,
        sample=1e-3,  # Downsampling of frequent words.
        window=context_size,
        workers=4,  # Number of threads to run in parallel.
        min_count=min_word_count
    )

    model.init_sims(replace=True)  # make the model memory-efficient.
    w2v_model_file_name = "./W2VModels/%s_W2V-%dfeatures_%dminwords_%dcontext%s.model"\
                              % (dataset, embedding_size, min_word_count, context_size, ("-extended" if extend_data else ""))
    model.save(w2v_model_file_name)  # save model.

    return model


def create_w2v_cluster(model, w2v_cluster_file_name, using_google):
    """
    Run the K-MEAN algorithm on the word embedings to cluster the data into #samples / 5 clusters.
    :param model: the word-to-vec model containing the word embeddings.
    :param w2v_cluster_file_name: file name to save the cluster.
    :param using_google: flag to know if the word embeddings are comming from google (ie: vocab of 3,000,000 words!)
    :return: dictionaries from word to cluster index, and from cluster index to word.
    """
    word_vectors = model.syn0
    if using_google:
        num_clusters = 100000  # 100,000 clusters of 3 million words = 30 words by cluster.
    else:
        num_clusters = word_vectors.shape[0] / 5  # amount of clusters such that each has 5 words on average.
    print "clustering %d word vectors into %d clusters..." % (word_vectors.shape[0], num_clusters)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, n_jobs=10)  # n_init = 10 random restarts, n_jobs = parallel on 10 CPUs.
    idx = kmeans.fit_predict(word_vectors)  # ordered list of centroids for each vocabulary word.
    print "done."
    print "Saving cluster to file..."
    word2centroid = dict(zip(model.index2word, idx))  # "word" -> cluster_index
    centroid2word = dict(zip(idx, model.index2word))  # cluster_index -> "word"
    data_file = open(w2v_cluster_file_name, 'wb')
    cPickle.dump((word2centroid, centroid2word), data_file, protocol=cPickle.HIGHEST_PROTOCOL)
    data_file.close()
    return word2centroid, centroid2word


# def review_to_embedding(review, model):
#     """
#     Compute the average vector embeddings for a review.
#     :param review: the review to get the embedding for = list of sentences, where each sentence is a list of words.
#     :param model: word2vec embeddings.
#     :return: a numpy array of average embeddings, as well as the number of skipped words and total words.
#     """
#     num_features = model.syn0.shape[1]
#     featureVec = np.zeros((num_features,), dtype="float32")
#     nwords = 0.  # total number of words in that review
#     model_vocab = set(model.index2word)  # model's vocabulary
#     nskipped = 0.
#     for sentence in review:
#         for word in sentence:
#             if word in model_vocab:
#                 featureVec = np.add(featureVec, model[word])  # sum all word embeddings.
#                 nwords += 1.
#             else:
#                 nskipped += 1.
#                 # print "%s not in W2V vocab!" % word
#     if nwords > 0:
#         featureVec = np.divide(featureVec, nwords)  # average all word embeddings.
# 
#     return featureVec, nskipped, nwords+nskipped


def review_to_boc(review, word2centroid):
    """
    Get the bag-of-centroid for a review. 
    :param review: the review to get the boc for = list of sentences, where each sentence is a list of words.
    :param word2centroid: dictionary from "word" to embedding centroid index.
    :return: a numpy array of centroid counts, as well as the number of skipped words and total words.
    """
    num_features = max(word2centroid.values()) + 1  # max centroid index +1(bcs zero-indexed)
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.  # total number of words in that review
    nskipped = 0.
    for sentence in review:
        for word in sentence:
            if word in word2centroid:
                idx = word2centroid[word]  # cluster(centroid) index.
                featureVec[idx] += 1.
                nwords += 1.
            else:
                nskipped += 1.
                # print "%s not in W2V vocab!" % word

    return featureVec, nskipped, nwords+nskipped


def random_forest_on_w2v(dataset, extend_data=False, embedding_size=300, forest_size=100):
    """
    Train a random forest classifier with Bag of W2V-Centroids features.
    :param dataset: string to decide which dataset to load (imdb or rot-tom).
    :param extend_data: flag to decide to load the extended data (imdb + rot-tom)
    :param embedding_size: int size of word embedding vector.
    :param forest_size: int number of random classification trees.
    :return: Nothing, writes to a file the predictions of the test set.
    """
    using_google = (embedding_size == -1)
    #############################
    # LOAD THE WORD-2-VEC MODEL #
    #############################
    if using_google:
        print "\nLoading Google's word embeddings..."
        w2v_model_file_name = "./W2VModels/GoogleNews-vectors-negative300.bin.gz"
        model = word2vec.Word2Vec.load_word2vec_format(w2v_model_file_name, binary=True)
    else:
        context_size = 10  # Context / window size. TODO: try more!
        min_word_count = 40  # Minimum word count. TODO: try less!
        w2v_model_file_name = "./W2VModels/%s_W2V-%dfeatures_%dminwords_%dcontext%s.model"\
                              % (dataset, embedding_size, min_word_count, context_size, ("-extended" if extend_data else ""))
        if os.path.isfile(w2v_model_file_name):
            print "\nLoading existing word embeddings..."
            model = word2vec.Word2Vec.load(w2v_model_file_name)
        else:
            model = create_w2v_model(dataset, extend_data, embedding_size, context_size, min_word_count)

    print "done."
    print "model shape:", model.syn0.shape

    ####################################
    # LOAD CLUSTERS OF WORD EMBEDDINGS #
    ####################################
    if using_google:
        w2v_cluster_file_name = "./W2VClusters/GoogleNews-vectors-negative300.cluster.pkl"
        if os.path.isfile(w2v_cluster_file_name):
            print "\nLoading Google's word embeddings' cluster..."
            word2centroid, centroid2word = cPickle.load(open(w2v_cluster_file_name, 'rb'))
        else:
            print "\nCreating Google's word embeddings' cluster..."
            word2centroid, centroid2word = create_w2v_cluster(model, w2v_cluster_file_name, using_google)
    else:
        w2v_cluster_file_name = "./W2VClusters/%s_W2V-%dfeatures_%dminwords_%dcontext%s.cluster.pkl"\
                              % (dataset, embedding_size, min_word_count, context_size, ("-extended" if extend_data else ""))
        if os.path.isfile(w2v_cluster_file_name):
            print "\nLoading existing word embeddings' cluster..."
            word2centroid, centroid2word = cPickle.load(open(w2v_cluster_file_name, 'rb'))
        else:
            print "\nCreating word embeddings' cluster..."
            word2centroid, centroid2word = create_w2v_cluster(model, w2v_cluster_file_name, using_google)

    print "done."
    num_features = max(word2centroid.values()) + 1  # max centroid index +1(bcs zero-indexed)
    print "number of features:", num_features

    ############################################
    # GET X_TRAIN & X_TEST: REVIEWS EMBEDDINGS #
    ############################################
    print "\nLoading 'cleaned' dataset..."
    remove_stop = True  # remove stop words here!!
    extract_tokens = True
    if dataset == 'imdb':
        data_dir = IMDB_DATA_DIR
    elif dataset == 'rottom':
        data_dir = ROTTOM_DATA_DIR
    else:
        print "ERROR: Unknown dataset %s" % dataset
        return

    data_file_name = data_dir + "/processed/data_extended%s_remove-stop%s_tokens%s.pkl" % (
        ("1" if extend_data else "0"),
        ("1" if remove_stop else "0"),
        ("1" if extract_tokens else "0")
    )
    data = cPickle.load(open(data_file_name, 'rb'))

    labeled_train, unlabeled_train, test = data[0], data[1], data[2]
    labeled_train_reviews = data[3]  # type = list of `reviews`( = list of `sentences`( = list of word tokens) )
    unlabeled_train_reviews = data[4]  # type = list of `reviews`( = list of `sentences`( = list of word tokens) )
    test_reviews = data[5]  # type = list of `reviews`( = list of `sentences`( = list of word tokens) )
    print "done."
    num_train_reviews = len(labeled_train_reviews)
    num_test_reviews = len(test_reviews)
    print "number of training reviews:", num_train_reviews
    print "number of testing reviews:", num_test_reviews

    print "\nGet training reviews' bag-of-centroid..."
    train_data_features = np.zeros(
        (num_train_reviews, num_features),
        dtype="float32"
    )
    nskipped, nwords = 0, 0
    for i, review in enumerate(labeled_train_reviews):
        boc, skip, total = review_to_boc(review, word2centroid)
        train_data_features[i] = boc
        nskipped += skip
        nwords += total
    print train_data_features
    print train_data_features.shape
    print "skipped %d/%d (%f%%) words" % (nskipped, nwords, nskipped*100./nwords)

    print "\nGet testing reviews' bag-of-centroid..."
    test_data_features = np.zeros(
        (num_test_reviews, num_features),
        dtype="float32"
    )
    nskipped, nwords = 0, 0
    for i, review in enumerate(test_reviews):
        boc, skip, total = review_to_boc(review, word2centroid)
        test_data_features[i] = boc
        nskipped += skip
        nwords += total
    print test_data_features
    print test_data_features.shape
    print "skipped %d/%d (%f%%) words." % (nskipped, nwords, nskipped*100./nwords)

    ##########################
    # Training Random-Forest #
    ##########################
    print "\nTraining Random Forest(%d) classifier..." % forest_size
    forest = RandomForestClassifier(n_estimators=forest_size)  # initialize with some trees
    if dataset == 'imdb':
        forest = forest.fit(train_data_features, labeled_train['sentiment'])
    elif dataset == 'rottom':
        forest = forest.fit(train_data_features, labeled_train['Sentiment'])
    else:
        print "ERROR: Unknown dataset %s" % dataset
        return
    print "done."

    train_predictions = forest.predict(train_data_features)
    if dataset == 'imdb':
        mse = mean_squared_error(labeled_train['sentiment'], train_predictions)
    elif dataset == 'rottom':
        mse = mean_squared_error(labeled_train['Sentiment'], train_predictions)
    else:
        print "ERROR: Unknown dataset %s" % dataset
        return
    print "TRAINING mse =", mse

    ##########################################################
    # Predicting on test set and make a csv submission file. #
    ##########################################################
    print "\nMaking predictions on test data..."
    test_predictions = forest.predict(test_data_features)
    print "done."

    print "\nWritting predictions to file..."
    if dataset == 'imdb':
        output = pd.DataFrame(data={"id": test['id'], "sentiment": test_predictions})
    elif dataset == 'rottom':
        output = pd.DataFrame(data={"PhraseId": test['PhraseId'], "Sentiment": test_predictions})
    else:
        print "ERROR: Unknown dataset %s" % dataset
        return

    w2v_model_file_name = w2v_model_file_name.replace('./W2VModels/'+dataset+'_', '')
    w2v_model_file_name = w2v_model_file_name.replace('.bin.gz', '')
    w2v_model_file_name = w2v_model_file_name.replace(('-extended' if extend_data else '')+'.model', '')
    file_name = "./ModelResponses/boc/%s_forest%d_%s_%spredictions.csv"\
                % (dataset, forest_size, w2v_model_file_name, ("extended_" if extend_data else ""))
    output.to_csv(file_name, index=False, quoting=3)
    print "done."


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() in ("yes", "true", "t", "1"))
    parser.add_argument(
        'dataset',
        choices=('imdb', 'rottom'),
        help='Dataset to load and predict on.'
    )
    parser.add_argument(
        '--features_size',
        type=int,
        default=300,
        help='Number of features per review = embedding size. if -1 will use pre-trained Google\'s embeddings of size 300.'
    )
    parser.add_argument(
        '--forest_size',
        type=int,
        default=100,
        help='Number of classification trees in Random Forest algorithm.'
    )
    parser.add_argument(
        '--extend',
        type='bool',
        default=False,
        help='Flag to decide to load the extended data (imdb + rot-tom) or not.'
    )
    args = parser.parse_args()
    print 'args:', args

    print "\nRunning RandomForest on Bag Of (Word2Vec) Clusters"
    random_forest_on_w2v(args.dataset, extend_data=args.extend, embedding_size=args.features_size, forest_size=args.forest_size)
