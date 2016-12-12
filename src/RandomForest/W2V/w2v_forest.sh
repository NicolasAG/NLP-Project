#!/usr/bin/env bash

python random_forest.py imdb --features_size 300 --forest_size 100 --extend True
python random_forest.py imdb --features_size 500 --forest_size 100 --extend True
# python random_forest.py imdb --features_size -1 --forest_size 100 --extend True
python random_forest.py imdb --features_size 300 --forest_size 500 --extend True
python random_forest.py imdb --features_size 500 --forest_size 500 --extend True
# python random_forest.py imdb --features_size -1 --forest_size 500 --extend True
python random_forest.py imdb --features_size 300 --forest_size 100 --extend False
python random_forest.py imdb --features_size 500 --forest_size 100 --extend False
# python random_forest.py imdb --features_size -1 --forest_size 100 --extend False
python random_forest.py imdb --features_size 300 --forest_size 500 --extend False
python random_forest.py imdb --features_size 500 --forest_size 500 --extend False
# python random_forest.py imdb --features_size -1 --forest_size 500 --extend False

python random_forest.py rottom --features_size 300 --forest_size 100 --extend True
python random_forest.py rottom --features_size 500 --forest_size 100 --extend True
# python random_forest.py rottom --features_size -1 --forest_size 100 --extend True
python random_forest.py rottom --features_size 300 --forest_size 500 --extend True
python random_forest.py rottom --features_size 500 --forest_size 500 --extend True
# python random_forest.py rottom --features_size -1 --forest_size 500 --extend True
python random_forest.py rottom --features_size 300 --forest_size 100 --extend False
python random_forest.py rottom --features_size 500 --forest_size 100 --extend False
# python random_forest.py rottom --features_size -1 --forest_size 100 --extend False
python random_forest.py rottom --features_size 300 --forest_size 500 --extend False
python random_forest.py rottom --features_size 500 --forest_size 500 --extend False
# python random_forest.py rottom --features_size -1 --forest_size 500 --extend False

