import numpy as np
import torch
import torch.nn as nn

import pprint
pp = pprint.PrettyPrinter()

input = torch.ones(2, 3, 4)
input[0, 1].multiply_(2)
pp.pprint(input)

linear = nn.Linear(4, 2)
linear_output = linear.forward(input)
pp.pprint(linear_output)