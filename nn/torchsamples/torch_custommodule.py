import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pprint
pp = pprint.PrettyPrinter()


class MultilayerPerceptron(nn.Module):

    def __init__(self, inputSize, hiddenSize):
        # Call to the __init__ function of the super class
        super(MultilayerPerceptron, self).__init__()

        # Bookkeeping: Saving the initialization parameters
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize

        # Defining of our layers
        self.linearFirst = nn.Linear(self.inputSize, self.hiddenSize)
        self.relu = nn.ReLU()
        self.linearSecond = nn.Linear(self.hiddenSize, self.inputSize)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        l1 = self.linearFirst(x)
        relu = self.relu(l1)
        l2 = self.linearSecond(relu)
        output = self.sigmoid(l2)
        return output


# Create our model
model = MultilayerPerceptron(5, 3)
pp.pprint(list(model.named_parameters()))
# pp.pprint(list(model.parameters()))

input = torch.ones(3, 5)
input[1].multiply_(2)

# Define loss using a predefined loss function
loss_function = nn.BCELoss()

# Calculate how our model is doing now
# pp.pprint(loss_function(output, output).item())

actual = (input + torch.ones_like(input))/4
output = model(input)

pp.pprint({
    'actual': actual,
    'output': output,
    'loss1': loss_function(input/4, actual),
    'loss2': loss_function(output, actual),
    'loss3': loss_function(actual, actual),
})

# Define the optimizer
sgd = optim.SGD(model.parameters(), lr=1e-1)

# Set the number of epoch, which determines the number of training iterations
n_epoch = 20

for epoch in range(n_epoch):
  # Set the gradients to 0
  sgd.zero_grad()

  # Get the model predictions
  output = model(input)

  # Get the loss
  loss = loss_function(output, actual)

  # Print stats
  pp.pprint({
      'Epoch': epoch,
      'loss': loss,
  })

  # Compute the gradients
  loss.backward()

  # Take a step to optimize the weights
  sgd.step()


output = model(input)
pp.pprint({
    'actual': actual,
    'output': output,
})



