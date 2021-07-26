import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial

import pprint
pp = pprint.PrettyPrinter()
np.set_printoptions(precision=4, suppress=True)

corpus = [
          "We always come to Paris",
          "The professor and his wife are from Australia",
          "I live in Stanford",
          "He comes from Taiwan",
          "The capital of Turkey is Ankara",
          "Paris is in France and is pretty cool"
         ]

locations = set(["australia", "ankara", "paris", "stanford", "taiwan", "turkey", "france", "spain", "canada", "california", "india"])


def get_word_to_ix(train_sentences):
    vocabulary = set(w for s in train_sentences for w in s)
    vocabulary.add("<unk>")
    vocabulary.add("<pad>")
    ix_to_word = sorted(list(vocabulary))
    return {word: ind for ind, word in enumerate(ix_to_word)}


def pad_window(sentence, window_size, pad_token="<pad>"):
    window = [pad_token] * window_size
    return window + sentence + window


def convert_tokens_to_indices(sentence, word_to_ix):
    return [word_to_ix.get(token, word_to_ix["<unk>"]) for token in sentence]


def custom_collate_fn(batch, window_size, word_to_ix):
    # Break our batch into the training examples (x) and labels (y)
    x, y = zip(*batch)

    # Pad the train examples.
    x = [pad_window(s, window_size=window_size) for s in x]

    # Now we need to turn words in our training examples to indices.
    x = [convert_tokens_to_indices(s, word_to_ix) for s in x]

    # We will now pad the examples so that the lengths of all the example in
    # one batch are the same, making it possible to do matrix operations.
    # We set the batch_first parameter to True so that the returned matrix has
    # the batch as the first dimension.
    pad_token_ix = word_to_ix["<pad>"]

    # pad_sequence function expects the input to be a tensor, so we turn x into one
    x = [torch.LongTensor(x_i) for x_i in x]
    x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=pad_token_ix)

    # We will also pad the labels. Before doing so, we will record the number
    # of labels so that we know how many words existed in each example.
    lengths = [len(label) for label in y]
    lengths = torch.LongTensor(lengths)

    y = [torch.LongTensor(y_i) for y_i in y]
    y_padded = nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)

    # We are now ready to return our variables. The order we return our variables
    # here will match the order we read them in our training loop.
    return x_padded, y_padded, lengths


class WordWindowClassifier(nn.Module):

    def __init__(self, hyperparameters, vocab_size, pad_ix=0):
        super(WordWindowClassifier, self).__init__()

        """ Instance variables """
        self.window_size = hyperparameters["window_size"]
        self.embed_dim = hyperparameters["embed_dim"]
        self.hidden_dim = hyperparameters["hidden_dim"]
        self.freeze_embeddings = hyperparameters["freeze_embeddings"]

        """ Embedding Layer 
        Takes in a tensor containing embedding indices, and returns the 
        corresponding embeddings. The output is of dim 
        (number_of_indices * embedding_dim).
    
        If freeze_embeddings is True, set the embedding layer parameters to be
        non-trainable. This is useful if we only want the parameters other than the
        embeddings parameters to change. 
    
        """
        self.embeds = nn.Embedding(vocab_size, self.embed_dim, padding_idx=pad_ix)
        if self.freeze_embeddings:
            self.embed_layer.weight.requires_grad = False

        """ Hidden Layer
        """
        full_window_size = 2 * self.window_size + 1
        self.hidden_layer = nn.Sequential(
            nn.Linear(full_window_size * self.embed_dim, self.hidden_dim),
            nn.Tanh()
        )

        """ Output Layer
        """
        self.output_layer = nn.Linear(self.hidden_dim, 1)

        """ Probabilities 
        """
        self.probabilities = nn.Sigmoid()

    def forward(self, inputs):
        """
        Let B:= batch_size
            L:= window-padded sentence length
            D:= self.embed_dim
            S:= self.window_size
            H:= self.hidden_dim

        inputs: a (B, L) tensor of token indices
        """
        B, L = inputs.size()

        """
        Reshaping.
        Takes in a (B, L) LongTensor
        Outputs a (B, L~, S) LongTensor
        """
        # Fist, get our word windows for each word in our input.
        token_windows = inputs.unfold(1, 2 * self.window_size + 1, 1)
        _, adjusted_length, _ = token_windows.size()

        # Good idea to do internal tensor-size sanity checks, at the least in comments!
        assert token_windows.size() == (B, adjusted_length, 2 * self.window_size + 1)

        """
        Embedding.
        Takes in a torch.LongTensor of size (B, L~, S) 
        Outputs a (B, L~, S, D) FloatTensor.
        """
        embedded_windows = self.embeds(token_windows)

        """
        Reshaping.
        Takes in a (B, L~, S, D) FloatTensor.
        Resizes it into a (B, L~, S*D) FloatTensor.
        -1 argument "infers" what the last dimension should be based on leftover axes.
        """
        embedded_windows = embedded_windows.view(B, adjusted_length, -1)

        """
        Layer 1.
        Takes in a (B, L~, S*D) FloatTensor.
        Resizes it into a (B, L~, H) FloatTensor
        """
        layer_1 = self.hidden_layer(embedded_windows)

        """
        Layer 2
        Takes in a (B, L~, H) FloatTensor.
        Resizes it into a (B, L~, 1) FloatTensor.
        """
        output = self.output_layer(layer_1)

        """
        Softmax.
        Takes in a (B, L~, 1) FloatTensor of unnormalized class scores.
        Outputs a (B, L~, 1) FloatTensor of (log-)normalized class scores.
        """
        output = self.probabilities(output)
        output = output.view(B, -1)

        return output


# Define a loss function, which computes to binary cross entropy loss
def loss_function(batch_outputs, batch_labels, batch_lengths):
    # Calculate the loss for the whole batch
    bceloss = nn.BCELoss()
    loss = bceloss(batch_outputs, batch_labels.float())

    # Rescale the loss. Remember that we have used lengths to store the
    # number of words in each training example
    loss = loss / batch_lengths.sum().float()

    return loss


# Function that will be called in every epoch
def train_epoch(loss_function, optimizer, model, loader):
    # Keep track of the total loss for the batch
    total_loss = 0
    for batch_inputs, batch_labels, batch_lengths in loader:
        optimizer.zero_grad()
        outputs = model.forward(batch_inputs)
        loss = loss_function(outputs, batch_labels, batch_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


# Function containing our main training loop
def train(loss_function, optimizer, model, loader, num_epochs=10000):
    for epoch in range(num_epochs):
        epoch_loss = train_epoch(loss_function, optimizer, model, loader)
        if epoch % 100 == 0:
            print(epoch_loss)


train_sentences = [sent.lower().split() for sent in corpus]
train_labels = [[1 if word in locations else 0 for word in sent] for sent in train_sentences]

word_to_ix = get_word_to_ix(train_sentences)

# Instantiate the DataLoader
data = list(zip(train_sentences, train_labels))
collate_fn = partial(custom_collate_fn, window_size=2, word_to_ix=word_to_ix)
loader = DataLoader(data, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Initialize the model
model_hyperparameters = {
    "window_size": 2,
    "embed_dim": 25,
    "hidden_dim": 25,
    "freeze_embeddings": False,
}

vocab_size = len(word_to_ix)
model = WordWindowClassifier(model_hyperparameters, vocab_size)

# Define optimizer
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train(loss_function, optimizer, model, loader)

test_corpus = [
          "She has been to Turkey",
          "Our house is in Spain",
          "We have never been to Canada",
          "Stanford is in California",
          "He comes from India"
         ]
test_sentences = [sent.lower().split() for sent in test_corpus]
test_labels = [[1 if word in locations else 0 for word in s] for s in test_sentences]
test_data = list(zip(test_sentences, test_labels))

inputs, labels, lengths = custom_collate_fn(test_data, window_size=2, word_to_ix=word_to_ix)
outputs = model.forward(inputs)
pp.pprint({
    'inputs': inputs,
    'outputs': outputs,
    'labels': labels,
    'outputs_dec': outputs.detach().numpy()
})
