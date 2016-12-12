import cPickle
import argparse

import numpy as np


def make_worddict(sentences):
    """
    Build a word dictionary to represent each word in some sentences as a number.
    :param sentences: array of sentences. each sentence is an array of word.
    :return: two dictionaries: one from word to number, the other from number to word.
    """
    wordcount = dict()
    for s in sentences:
        for w in s:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    words = wordcount.keys()

    counts = wordcount.values()
    sorted_idx = np.argsort(counts)[::-1]

    word2idx = {'UNK':0}
    idx2word = {0:'UNK'}
    for i, idx in enumerate(sorted_idx):
        word2idx[words[idx]] = i+1  # leave 0 for unseen words (UNK)
        idx2word[i+1] = words[idx]

    print "%d total words. %d unique words." % (np.sum(counts), len(words))
    return word2idx, idx2word


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() in ("yes", "true", "t", "1"))
    parser.add_argument(
        '--extend_data',
        type='bool',
        default=False,
        help='flag to decide to loaded the extended version (with rot-tom) of the data.'
    )
    parser.add_argument(
        '--remove_stop',
        type='bool',
        default=True,
        help='flag to decide to remove stop words and numbers or not.'
    )
    args = parser.parse_args()
    print 'args:', args


    ###
    # LOAD DATA
    ###
    print "\nLoading dataset..."
    data_file_name = "./data_extended%s_remove-stop%s_tokens1.pkl" % (
        ("1" if args.extend_data else "0"),
        ("1" if args.remove_stop else "0")
    )
    data = cPickle.load(open(data_file_name, 'rb'))

    labeled_train_reviews = data[3]  # type = list of `reviews`( = list of `sentences`( = list of word tokens) )
    unlabeled_train_reviews = data[4]  # type = list of `reviews`( = list of `sentences`( = list of word tokens) )
    test_reviews = data[5]  # type = list of `reviews`( = list of `sentences`( = list of word tokens) )

    ###
    # BUILD WORD DICTIONARY on labeled and unlabeled data
    ###
    sentences = []  # store all sentences from all train reviews
    for review in labeled_train_reviews:
        sentences += review
    for review in unlabeled_train_reviews:
        sentences += review
    print "number of train sentences %d" % len(sentences)

    print "\nBuilding dictionary..."
    word2idx, idx2word = make_worddict(sentences)
    f = open(
        "./dicts_extended%s_remove-stop%s.pkl" % (
            ("1" if args.extend_data else "0"),
            ("1" if args.remove_stop else "0")
        ),
        "wb"
    )
    cPickle.dump((word2idx, idx2word), f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print "done."

    ###
    # CONVERT DATA INTO NUMBERS
    ###
    print "\nConverting word data into number data..."
    print "labeled train..."
    new_labeled_train_reviews = []  # list of reviews. each review is list of numbers.
    for review in labeled_train_reviews:
        words = []  # list of numbers.
        for sentence in review:
            for w in sentence:
                words.append(word2idx[w])
        new_labeled_train_reviews.append(np.asarray(words))
    print "unlabeled train..."
    new_unlabeled_train_reviews = []  # list of reviews. each review is list of numbers.
    for review in unlabeled_train_reviews:
        words = []  # list of numbers.
        for sentence in review:
            for w in sentence:
                words.append(word2idx[w])
        new_unlabeled_train_reviews.append(np.asarray(words))
    print "test..."
    new_test_reviews = []  # list of reviews. each review is list of numbers.
    for review in test_reviews:
        words = []  # list of numbers.
        for sentence in review:
            for w in sentence:
                words.append(word2idx[w] if w in word2idx else 0)  # unseen words = 0
        new_test_reviews.append(np.asarray(words))

    data = (data[0], data[1], data[2], new_labeled_train_reviews, new_unlabeled_train_reviews, new_test_reviews)
    f = open(
        "./ndata_extended%s_remove-stop%s.pkl" % (
            ("1" if args.extend_data else "0"),
            ("1" if args.remove_stop else "0")
        ),
        "wb"
    )
    cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print "done."

