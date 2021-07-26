import numpy as np
import torch
import torch.nn as nn

import pprint
pp = pprint.PrettyPrinter()

x = torch.tensor([
    3
], requires_grad=True, dtype=float)

y = 2 * x * x
y = y * y
y.backward()

pp.pprint(y)
pp.pprint(x)
pp.pprint(x.grad)


if 0:
    y = y - 10
    y.backward()
    pp.pprint(y)
    pp.pprint(x)
    pp.pprint(x.grad)

