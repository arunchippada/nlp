import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
import random
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.test.utils import datapath


def load_embedding_model():
    """ Load GloVe Vectors
        Return:
            wv_from_bin: All 400000 embeddings, each lengh 200
    """
    wv_from_bin = api.load("glove-wiki-gigaword-200")
    print("Loaded vocab size %i" % len(wv_from_bin.vocab.keys()))
    return wv_from_bin


def select_vectors(model, required_words, n=10000):
    words = list(model.vocab.keys())
    print("Shuffling words ...")
    random.seed(224)
    random.shuffle(words)
    words = words[:n]

    df = pd.DataFrame(columns=range(0, 200))
    for word in words:
        df.loc[word] = model.word_vec(word)

    for word in required_words:
        df.loc[word] = model.word_vec(word)

    return df


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


word_vec_from_csv = 1
reduced_from_csv = 1
vec_csv_file = "E:\\scripts\\python\\nlp\\nn\\word_vector\\glove_vec.csv"
red_csv_file = "E:\\scripts\\python\\nlp\\nn\\word_vector\\glove_reduced.csv"
req_words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'iraq']

if reduced_from_csv:
    reduced = pd.read_csv(red_csv_file, index_col=0)
else:
    if word_vec_from_csv:
        word_vectors = pd.read_csv(vec_csv_file, index_col=0)
    else:
        model = load_embedding_model()
        word_vectors = select_vectors(model, req_words)
        print("Writing vectors to csv")
        word_vectors.to_csv(vec_csv_file)

    reduced = reduce_dim(word_vectors)
    reduced.to_csv(red_csv_file)

vectors = reduced.loc[req_words]
plot_vectors(vectors)

# normalizing vectors to unit length, so that closeness of two vectors is directional closeness
lengths = np.linalg.norm(vectors, axis=1)
norm_vectors = vectors/np.array([lengths]).transpose()
plot_vectors(norm_vectors, fig=2)

plt.show()

