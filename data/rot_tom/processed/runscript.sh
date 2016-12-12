#!/usr/bin/env bash

python preprocess_data.py --extend_data True --remove_stop True --extract_tokens True
python preprocess_data.py --extend_data True --remove_stop True --extract_tokens False
python preprocess_data.py --extend_data True --remove_stop False --extract_tokens True
# python preprocess_data.py --extend_data True --remove_stop False --extract_tokens False
python preprocess_data.py --extend_data False --remove_stop True --extract_tokens True
python preprocess_data.py --extend_data False --remove_stop True --extract_tokens False
python preprocess_data.py --extend_data False --remove_stop False --extract_tokens True
# python preprocess_data.py --extend_data False --remove_stop False --extract_tokens False

python make_dictionary.py --extend_data False --remove_stop False
python make_dictionary.py --extend_data False --remove_stop True
python make_dictionary.py --extend_data True --remove_stop False
python make_dictionary.py --extend_data True --remove_stop True

