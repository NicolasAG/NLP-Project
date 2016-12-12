import argparse
import re
import pandas as pd
import warnings
import cPickle

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


STOP_LIST = list(stopwords.words("english"))
IMDB_DATA_DIR = "../../imdb"
ROTTOM_DATA_DIR = ".."


def clean_review(review, remove_stop=True, extract_tokens=False):
    """
    Remove stop words, remove html tags, remove punctuation, and put to lower case.
    :param review: the original review (type: string)
    :param remove_stop: flag to decide to remove stop words and numbers or not.
    :param extract_tokens: flag to decide to extract lists of word tokens or keep review as 1 string.
    :return: the clean version of the review: either as a string or as a list of list of tokens.
    """
    # TODO: be more clever with punctuation? :(=0, :)=1, ...

    sentences = sent_tokenize(review.decode('utf8').strip())  # get list of sentences
    for i, sentence in enumerate(sentences):
        warnings.filterwarnings('error')
        try:
            sentence = BeautifulSoup(sentence, 'html.parser').get_text()  # remove html tags
        except UserWarning:
            # print "User Warning on %s" % sentence
            pass
        sentence = re.sub(
            pattern="[^a-zA-Z0-9]",  # pattern to search for: everything except [a-zA-z0-9] so keep letters and numbers.
            repl=" ",  # pattern to replace it with
            string=sentence  # text to search in
        )
        sentence = sentence.lower()  # put all to lower case
        if remove_stop:
            sentence = re.sub("[0-9]+", "<#>", sentence)  # replace numbers by "<#>"
            sentence = word_tokenize(sentence.strip())  # split into array of words
            sentence = filter(lambda w: w not in STOP_LIST, sentence)  # remove stop words
            if not extract_tokens:
                sentence = ' '.join(sentence)  # join words of same sentence.
        else:
            if extract_tokens:
                sentence = word_tokenize(sentence.strip())  # split sentence into words
        # sentence is either a list of words(tokens) or a full string.
        sentences[i] = sentence

    if extract_tokens:
        return sentences
    else:
        return ' '.join(sentences)


def load_data(extend_data=False, remove_stop=True, extract_tokens=False):
    """
    Load ROTTEN TOMATOES dataset and preprocess reviews.
    :param extend_data: flag to decide to load the extended version (with imdb) of the data.
    :param remove_stop: flag to decide to remove stop words and numbers or not.
    :param extract_tokens: flag to decide to extract lists of word tokens or keep review as 1 string.
    :return: tuple of 6 elements: (train, unlabeled train=None, test) original and processed data.
    """
    print "\nLoading rotten tomatoes dataset..."
    # header row is the 0th one, columns delimited by '\t', ignore double quotes
    if extend_data:
        train = pd.read_csv(ROTTOM_DATA_DIR + '/train_extended.tsv', header=0, delimiter='\t', quoting=3)
    else:
        train = pd.read_csv(ROTTOM_DATA_DIR + '/train.tsv', header=0, delimiter='\t', quoting=3)
    test = pd.read_csv(ROTTOM_DATA_DIR + '/test.tsv', header=0, delimiter='\t', quoting=3)
    print "done."
    print "train:", train.shape, "with", train.columns.values
    print "test:", test.shape, "with", test.columns.values

    train_reviews = []
    test_reviews = []
    print "\nParsing train reviews..."
    for review in train['Phrase']:
        train_reviews.append(clean_review(review, remove_stop, extract_tokens))
    print "parsing test reviews..."
    for review in test['Phrase']:
        test_reviews.append(clean_review(review, remove_stop, extract_tokens))

    assert len(train_reviews) == len(train['Phrase'])
    assert len(test_reviews) == len(test['Phrase'])
    print "done."

    print train['Phrase'][0]
    print train_reviews[0]
    print ""
    print train['Phrase'][99359]
    print train_reviews[99359]

    return (
        train, None, test,
        train_reviews, None, test_reviews
    )

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
    parser.add_argument(
        '--extract_tokens',
        type='bool',
        default=False,
        help='flag to decide to extract lists of word tokens or keep reviews as 1 string.'
    )
    args = parser.parse_args()
    print 'args:', args

    data = load_data(args.extend_data, args.remove_stop, args.extract_tokens)
    file_name = "data_extended%s_remove-stop%s_tokens%s.pkl" % (
        ("1" if args.extend_data else "0"),
        ("1" if args.remove_stop else "0"),
        ("1" if args.extract_tokens else "0")
    )
    cPickle.dump(data, open(file_name, 'wb'))

