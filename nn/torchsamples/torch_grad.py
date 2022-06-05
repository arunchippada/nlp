import numpy as np
import torch
import torch.nn as nn

import pprint
pp = pprint.PrettyPrinter()

x = torch.tensor([
    3
], requires_grad=True, dtype=float)

z = 3 * x * x * x
z.backward()

pp.pprint(z)
pp.pprint(x)
pp.pprint(x.grad)

# When y.backward is called below, it computes the gradient at x in the y path and then adds to the existing gradient
# obtained from the z path. That is, gradient gets accumulated for x from multiple paths where it receives the gradient
# if x.grad is set to none, then only gradient from y path will be in x.grad below
# x.grad = None

y = 2 * x * x
# the y on RHS is treated as different variable, from the LHS y
y = y * y

# retain graph does not free up the variables and allows us to run backward again
y.backward(retain_graph=True)

pp.pprint(y)
pp.pprint(x)
pp.pprint(x.grad)

# assigning a new value to x is akin to creating a new variable. The earlier structure between x and y is not applicable
# for this new variable
x = torch.tensor([
    10
], requires_grad=True, dtype=float)

y.backward(retain_graph=True)

pp.pprint(y)
pp.pprint(x)
pp.pprint(x.grad)


if 0:
    y = y - 10
    y.backward()
    pp.pprint(y)
    pp.pprint(x)
    pp.pprint(x.grad)

