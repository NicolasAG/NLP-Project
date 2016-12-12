#!/usr/bin/env bash

THEANO_FLAGS='floatX=float32,device=gpu0' python lstm.py 'rottom' --optimizer 'adam' --extend_data False --feature_size 300
THEANO_FLAGS='floatX=float32,device=gpu0' python lstm.py 'rottom' --optimizer 'adam' --extend_data False --feature_size 500
THEANO_FLAGS='floatX=float32,device=gpu0' python lstm.py 'rottom' --optimizer 'adam' --extend_data True --feature_size 300
THEANO_FLAGS='floatX=float32,device=gpu0' python lstm.py 'rottom' --optimizer 'adam' --extend_data True --feature_size 500

THEANO_FLAGS='floatX=float32,device=gpu0' python lstm.py 'imdb' --optimizer 'adam' --extend_data False --valid_freq 100 --feature_size 300
THEANO_FLAGS='floatX=float32,device=gpu0' python lstm.py 'imdb' --optimizer 'adam' --extend_data False --valid_freq 100 --feature_size 500
THEANO_FLAGS='floatX=float32,device=gpu0' python lstm.py 'imdb' --optimizer 'adam' --extend_data True --valid_freq 100 --feature_size 300
THEANO_FLAGS='floatX=float32,device=gpu0' python lstm.py 'imdb' --optimizer 'adam' --extend_data True --valid_freq 100 --feature_size 500

