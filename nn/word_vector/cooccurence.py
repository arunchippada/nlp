import nltk
import pprint
import numpy as np
import pandas as pd
from nltk.corpus import reuters

START_TOKEN = '<START>'
END_TOKEN = '<END>'


def read_corpus(category="crude"):
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in reuters.words(f)] + [END_TOKEN] for f in files]


if 0:
    docs = read_corpus()
    pprint.pprint(docs[:3], compact=True, width=100)

# distinct words
s = pd.Series([w.lower() for w in reuters.words(categories='crude')])
tokens = [START_TOKEN] + list(s.value_counts().index) + [END_TOKEN]
print(tokens)


# cooccurennce = pd.DataFrame()

# print(cooccurennce)

# for d in read_corpus():
#    for i in range(0, len(d) - 1):
