# Preprocess Data

Preprocess the data in various ways using `preprocess_data.py`

--extend_data  Decide to load the extended version of the dataset or not (default=False)

--remove_stop  Decide to remove STOP WORDS from each review (default=True)

--extract_tokens  Decide of the output format:
if False, the data is just an array of long review strings. ex: ["review1", "review2", ...]
if True, the data is an array of reviews where each review is an array of sentences where each sentence is an array of words. ex: [ [["first","sentence"],["second","sentence"],["of","first","review"]], ... ]


Build word dictionaries using `make_dictionary.py`

--extend_data  Decide to load the extended version of the dataset or not (default=False)

--remove_stop  Decide to remove STOP WORDS from each review (default=True)

The resulting data will be an array of reviews where each review is an array of sentences where each sentence is an array of integers.
Each integer has a direct mapping to one word.
The dictionaries mapping words to integers and integers to words are stored as pickle files: `./dicts_extended<0|1>_remove-stop<0|1>.pkl`

