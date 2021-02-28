import gensim.downloader as api
import pprint

def load_embedding_model():
    """ Load GloVe Vectors
        Return:
            wv_from_bin: All 400000 embeddings, each lengh 200
    """
    wv_from_bin = api.load("glove-wiki-gigaword-200")
    return wv_from_bin


model = load_embedding_model()

# most similar
if 0:
    words = list(model.vocab.keys())
    sel_words = words[:1000]

    # time (1) sequential, series, order (2) togetherness
    # time = [('when', 0.8192390203475952), ('before', 0.8063887357711792), ('.', 0.7924259901046753), ('but', 0.7923368215560913), ('only', 0.782798171043396), ('same', 0.7813145518302917), ('just', 0.7760949730873108), ('one', 0.7725858688354492), ('after', 0.7658392190933228), ('even', 0.7655478119850159)]

    # last (1) end of the line (2) previous (3) endless
    # last = [('month', 0.9006603360176086), ('week', 0.8998246192932129), ('ago', 0.878319501876831), ('year', 0.8718104958534241), ('earlier', 0.8489571213722229), ('after', 0.8343580365180969), ('came', 0.8310091495513916), ('months', 0.8243457078933716), ('since', 0.8195637464523315), ('weeks', 0.8154026865959167)]

    # leave (1) move oneself away 2) let go
    # leave = [('stay', 0.8190094232559204), ('leaving', 0.8084293603897095), ('go', 0.7560037970542908), ('take', 0.7494380474090576), ('return', 0.7350345253944397), ('come', 0.7288329601287842), ('wait', 0.718921422958374), ('would', 0.712296724319458), ('rest', 0.7114236354827881), ('if', 0.7100386023521423)]

    for word in sel_words:
        most_similar = model.most_similar(word)
        print(word, "=", most_similar)


# antonym with more similarity
# distance leave stay = 0.18099063634872437
# distance leave go = 0.24399620294570923
if 0:
    words = ["happy", "leave"]

    for word in words:
        most_similar = model.most_similar(word)
        for (w2, similarity) in most_similar:
            print(f'distance {word} {w2} = {model.distance(word, w2)}')

    # distance happy sad = 0.4040136933326721
    # distance happy cheerful = 0.5172466933727264
    word = "happy"
    w2 = "sad"
    print(f'distance {word} {w2} = {model.distance(word, w2)}')
    w2 = "cheerful"
    print(f'distance {word} {w2} = {model.distance(word, w2)}')

# analogy - finds vectors that are close to a given vector (which is an arithmetic on the positive and negative vectors)
# Removing the directionality and considering only the angular distance (cosine distance) is applicable when we are
# interested in the closeness to a given vector
most_similar = model.most_similar(positive=['woman', 'king'], negative=['man']);
pprint.pprint(most_similar)

# daughter
most_similar = model.most_similar(positive=['woman', 'son'], negative=['man']);
pprint.pprint(most_similar)

# ships is the first. sea is 7th in this list
most_similar = model.most_similar(positive=['ship', 'road'], negative=['car']);
pprint.pprint(most_similar)

# hear is 2nd
most_similar = model.most_similar(positive=['ear', 'see'], negative=['eye']);
pprint.pprint(most_similar)

# bias
most_similar = model.most_similar(positive=['man', 'worker'], negative=['woman']);
pprint.pprint(most_similar)

most_similar = model.most_similar(positive=['woman', 'worker'], negative=['man']);
pprint.pprint(most_similar)

positive_words = ['white', 'worker']
negative_words = ['black']
most_similar = model.most_similar(positive=positive_words, negative=negative_words);
print(f"positive {positive_words} {negative_words}")
pprint.pprint(most_similar)

# immigrant, migrant, unemployed
positive_words = ['black', 'worker']
negative_words = ['white']
most_similar = model.most_similar(positive=positive_words, negative=negative_words);
print(f"positive {positive_words} {negative_words}")
pprint.pprint(most_similar)

# football, baseball
positive_words = ['man', 'player']
negative_words = ['woman']
most_similar = model.most_similar(positive=positive_words, negative=negative_words);
print(f"positive {positive_words} {negative_words}")
pprint.pprint(most_similar)

# athlete, professional, golfer, basketball
positive_words = ['woman', 'player']
negative_words = ['man']
most_similar = model.most_similar(positive=positive_words, negative=negative_words);
print(f"positive {positive_words} {negative_words}")
pprint.pprint(most_similar)
