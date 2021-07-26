import numpy as np
import torch
import torch.nn as nn

import pprint
pp = pprint.PrettyPrinter()
np.set_printoptions(precision=4, suppress=True)

# from list
a = [
    [0.1e-2, 0.2, 0.3, 0.4],
    [8e-1, 9e-3, 10, 11],
    [5, 6, 7, 4e-4]
]

n = np.array(a)
print(n)
pp.pprint(n)
pp.pprint(n.shape)

t = torch.tensor(a)
t1 = (t/100).numpy()

pp.pprint(t1)

t = torch.tensor(n)
pp.pprint(t)
pp.pprint(t.shape)

b = np.array([
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1]
])

# numpy matmul
c = np.matmul(n, b)
pp.pprint(c)
pp.pprint(c.shape)

# torch matmul
a = torch.ones((4, 3)) * 6
b = torch.ones(3) * 2
c = a @ b.T
pp.pprint(c)
pp.pprint(c.shape)

