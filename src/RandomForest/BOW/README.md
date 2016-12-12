# Bag of Word features

First, you need to create the directory where prediction files (.csv) will be saved:
`mkdir ModelResponses`

Then build some predictions using `python random_forest.py <dataset>`

with the following flags:

- `<dataset>`  must be either 'imdb' or 'rottom'.
- --feature_size  vocabulary size = number of features per review (default=5000).
- --forest_size  number of decision trees in the random forest algorithm (default=100).
- --extend  either 'true' or 'false' to know if using the extended version of the dataset (default=False).

