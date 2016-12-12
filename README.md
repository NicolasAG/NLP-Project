# NLP-Project - Sentiment Analysis on Movie Reviews

Text Classification task.

## Data

IMDB and Rotten Tomatoes reviews.

inspired by two kaggle competitions:

- https://www.kaggle.com/c/word2vec-nlp-tutorial
- https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews

## Methods
1) Random Forest

- BoW features
- W2V features


2) LSTM Recurrent Neural Network

## Setup

### Data setup

####IMDB

1) Combine IMDB reviews with Rotten-Tomatoes reviews:
```
cd data/imdb/
python combine_rottom.py
```
2) Preprocess IMDB dataset in many ways. Save all versions in `./processed/` directory
```
cd ./processed/
./runscript.sh
```

####Rotten Tomatoes
1) Combine Rotten-Tomatoes reviews with IMDB reviews:
```
cd data/rot_tom/
python combine_imdb.py
```
2) Preprocess Rotten-Tomatoes dataset in many ways. Save all versions in `./processed/` directory
```
cd ./processed/
./runscript.sh
```

### Random Forest

#### BoW features
```
cd src/RandomForest/BOW/
mkdir ModelResponses
./bow_forest.sh
```

#### W2V features
Average word embeddings to get review embedding features:
```
cd src/RandomForest/W2V/
mkdir ModelResponses
./w2v_forest.sh
```

Cluster word embeddings to get bag-of-centroids features:
```
cd src/RandomForest/W2V/
mkdir ModelResponses
mkdir ModelResponses/boc
mkdir W2VClusters
./boc_forest.sh
```

### LSTM RNN
```
cd src/RNN/
mkdir ModelResponses
mkdir LSTM-Models
./runscript.sh
```

## python requirements
theano
lasagne
sklearn
nltk
numpy
cPickle
pandas
gensim


