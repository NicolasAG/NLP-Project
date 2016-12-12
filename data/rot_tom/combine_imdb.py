import pandas as pd


def main():
    print "\nLoading rotten tomatoes dataset..."
    # header row is the 0th one, columns delimited by '\t', ignore double quotes
    rottom_train = pd.read_csv('./train.tsv', header=0, delimiter='\t', quoting=3)
    print "done."
    print "rot-tom train:", rottom_train.shape, "whith", rottom_train.columns.values

    print "\nLoading imdb dataset..."
    imdb_train = pd.read_csv('../imdb/labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
    print "done."
    print "imdb train:", imdb_train.shape, "with", imdb_train.columns.values

    print "\nTransforming rotten tommatoes sentiments..."
    phrases = []
    sentiments = []
    ids = []
    for row in imdb_train.itertuples():  # for each review,
        # print row
        # index = row[0]
        review_id = row[1]
        sent = row[2]
        review = row[3]
        if sent == 0:
            ids.append(str(review_id))
            phrases.append(review)
            sentiments.append(0)
            # sentiments.append(1)
        elif sent == 1:
            ids.append(str(review_id))
            phrases.append(review)
            sentiments.append(4)
            # sentiments.append(3)
        else:
            print "ERROR: this should never happen! Sentiment = %d" % sent
    assert len(ids) == len(phrases) == len(sentiments)
    print "done."
    print "added %d" % len(ids)

    print "\nSaving combination to file..."
    new_frames = pd.DataFrame(
        data={
            "PhraseId": ids,
            "SentenceId": ids,
            "Phrase": phrases,
            "Sentiment": sentiments
        },
        columns=["PhraseId", "SentenceId", "Phrase", "Sentiment"]
    )
    output = pd.concat([rottom_train, new_frames])
    output.to_csv("./train_extended.tsv", sep='\t', index=False, quoting=3, quotechar='')
    print "done."


if __name__ == '__main__':
    main()

