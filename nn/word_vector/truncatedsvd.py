import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import reuters
from sklearn.decomposition import TruncatedSVD

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


def reduce_dim(df, k=2):
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    matrix = df.to_numpy()

    print("Running Truncated SVD over %i words..." % (matrix.shape[0]))
    svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=42)
    matrix_reduced = svd.fit_transform(matrix)
    print("Done.")

    reduced_df = pd.DataFrame(matrix_reduced, index=df.index, columns=['x', 'y'])
    return reduced_df


def plot_vectors(df, fig=1):
    plt.figure(fig)
    sns.scatterplot(x='x', y='y', data=df, label="Word vectors")
    ax = plt.gca()
    for w in df.index:
        ax.text(df.loc[w]['x'], df.loc[w]['y'], w)


cooccurence_from_csv = 1
reduced_from_csv = 1
co_csv_file = "E:\\scripts\\python\\nlp\\nn\\word_vector\\reuters_cooccurence.csv"
red_csv_file = "E:\\scripts\\python\\nlp\\nn\\word_vector\\reuters_reduced.csv"

test_corpus = ["{} The quick brown fox jumped over the lazy dog {}".format(START_TOKEN, END_TOKEN).split(" "),
               "{} The quick yellow bobcat jumped over the wooden fence {}".format(START_TOKEN, END_TOKEN).split(
                   " ")]

if reduced_from_csv:
    reduced = pd.read_csv(red_csv_file, index_col=0)
else:
    if cooccurence_from_csv:
        cooccurence = pd.read_csv(co_csv_file, index_col=0)
    else:
        # cooccurence = get_cooccurence(test_corpus, 1)
        cooccurence = get_cooccurence(read_corpus())
        print("Writing cooccurence to csv")
        cooccurence.to_csv(co_csv_file)

    reduced = reduce_dim(cooccurence)
    reduced.to_csv(red_csv_file)

words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'iraq']
vectors = reduced.loc[words]
plot_vectors(vectors)

# normalizing vectors to unit length, so that closeness of two vectors is directional closeness
# if two words have similar coocurrence - have similar ratios of occurence for each center word
# but are different in terms of overall frequency (i.e. one word appears more frequently than another across the
# documents), the normalization will remove the affect of the frequency
lengths = np.linalg.norm(vectors, axis=1)

norm_vectors = vectors/np.array([lengths]).transpose()
print(norm_vectors)

plot_vectors(norm_vectors, fig=2)

plt.show()






