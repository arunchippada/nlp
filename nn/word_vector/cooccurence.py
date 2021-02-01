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


def get_cooccurence(docs, w=4):
    # distinct words
    s = {w.lower() for w in [t for doc in docs for t in doc]}
    tokens = sorted(list(s))
    num_tokens = len(tokens)

    cooccurennce = pd.DataFrame(np.zeros((num_tokens, num_tokens), dtype=int), index=tokens, columns=tokens)

    for doc in docs:
        docLength = len(doc)
        for i in range(0, docLength):
            vw = doc[i].lower()
            for j in range(i - w, i + w + 1):
                if i != j and 0 <= j < docLength:
                    ow = doc[j].lower()
                    cooccurennce.loc[vw][ow] = cooccurennce.loc[vw][ow] + 1

    return cooccurennce

if 0:
    docs = read_corpus()
    pprint.pprint(docs[:3], compact=True, width=100)

test_corpus = ["{} All that glitters isn't gold {}".format(START_TOKEN, END_TOKEN).split(" "), "{} All's well that ends well {}".format(START_TOKEN, END_TOKEN).split(" ")]
test_cooccurence = get_cooccurence(test_corpus, 1)

print(test_cooccurence)
print(test_cooccurence.to_numpy())

# print(get_cooccurence(read_corpus()))





