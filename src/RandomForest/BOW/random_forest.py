import argparse
import pandas as pd
import cPickle
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error


IMDB_DATA_DIR = "../../../data/imdb"
ROTTOM_DATA_DIR = "../../../data/rot_tom"


def random_forest_on_bow(dataset, extend_data=False, vocab_limit=5000, forest_size=100):
    """
    Train a random forest classifier with Bag-of-Words features.
    :param dataset: string to decide which dataset to load (imdb or rot-tom).
    :param extend_data: flag to decide to load the extended data (imdb + rot-tom)
    :param vocab_limit: int size of vocabulary (number of features).
    :param forest_size: int number of random classification trees.
    :return: Nothing, writes to a file the predictions of the test set.
    """
    assert dataset in ['imdb', 'rottom']
    print "\nLoading dataset..."
    remove_stop = True
    extract_tokens = False
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

    # No use of unlabeled_ data for BOW.
    labeled_train, unlabeled_train, test = data[0], data[1], data[2]
    labeled_train_reviews = data[3]  # type = list of strings
    unlabeled_train_reviews = data[4]  # type = list of strings
    test_reviews = data[5]  # type = list of strings

    ##################################
    # Building BAG-OF-WORDS features #
    ##################################
    print "\nCreating the bag of words..."
    vectorizer = CountVectorizer(max_features=vocab_limit)
    # first: fit model and learn vocab; second: transforms training data into feature vectors.
    train_data_features = vectorizer.fit_transform(labeled_train_reviews)
    train_data_features = train_data_features.toarray()
    print train_data_features
    print train_data_features.shape

    # Array of vocabulary words.
    vocab = vectorizer.get_feature_names()
    print "Vocabulary size: %d" % len(vocab)

    # Array of counts of each vocabulary word.
    dist = np.sum(train_data_features, axis=0)
    # for tag, count in zip(vocab, dist):
    #     print tag, count

    ##########################
    # Training Random-Forest #
    ##########################
    print "\nTraining Random Forest(%d) classifier..." % forest_size
    forest = RandomForestClassifier(n_estimators=forest_size)  # initialize with 100 trees
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
    print "\nFitting test data..."
    test_data_features = vectorizer.transform(test_reviews)
    test_data_features = test_data_features.toarray()
    print test_data_features
    print test_data_features.shape

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
    file_name = "./ModelResponses/%s_forest%d_bow%d_%spredictions.csv"\
                % (dataset, forest_size, vocab_limit, ("extended_" if extend_data else ""))
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
        default=5000,
        help='Number of features per review = vocab size.'
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

    print "\nRunning RandomForest on BagOfWords features"
    random_forest_on_bow(args.dataset, extend_data=args.extend, vocab_limit=args.features_size, forest_size=args.forest_size)

