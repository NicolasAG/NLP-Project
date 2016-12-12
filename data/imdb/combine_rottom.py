import pandas as pd


def main():
    print "\nLoading imdb dataset..."
    # header row is the 0th one, columns delimited by '\t', ignore double quotes
    imdb_train = pd.read_csv('./labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
    print "done."
    print "imdb train:", imdb_train.shape, "with", imdb_train.columns.values

    print "\nLoading rotten tomatoes dataset..."
    rottom_train = pd.read_csv('../rot_tom/train.tsv', header=0, delimiter='\t', quoting=3)
    print "done."
    print "rot-tom train:", rottom_train.shape, "whith", rottom_train.columns.values

    print "\nTransforming rotten tommatoes sentiments..."
    reviews = []
    sentiments = []
    ids = []
    current_sentence = -1
    skipped = 0
    for row in rottom_train.itertuples():  # for each phrase: grab the 1st one that belongs to a sentence: ie: the sentence itself
        # index = row[0]
        phrase_id = row[1]
        sentence_id = row[2]
        phrase = row[3]
        sent = row[4]
        if sentence_id != current_sentence:  # if new sentence, grab this 1st phrase and its sentiment.
            current_sentence = sentence_id
            if sent > 2:
                ids.append("\""+str(phrase_id)+"\"")
                reviews.append("\""+phrase+"\"")
                sentiments.append(1)
            elif sent < 2:
                ids.append("\""+str(phrase_id)+"\"")
                reviews.append("\""+phrase+"\"")
                sentiments.append(0)
            else:
                skipped += 1
        else:
            skipped += 1
    assert len(ids) == len(reviews) == len(sentiments)
    print "done."
    print "added %d" % len(ids)
    print "skipped %d" % skipped

    print "\nSaving combination to file..."
    new_frames = pd.DataFrame(
        data={
            "id": ids,
            "sentiment": sentiments,
            "review": reviews
        },
        columns=["id", "sentiment", "review"]
    )
    output = pd.concat([imdb_train, new_frames])
    output.to_csv("./labeledTrainData_extended.tsv", sep='\t', index=False, quoting=3, quotechar='')
    print "done."


if __name__ == '__main__':
    main()

