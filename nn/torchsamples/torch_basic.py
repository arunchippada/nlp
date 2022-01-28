import numpy as np
import torch

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


list1 = [1]
a = np.array(list1)
t = torch.tensor(list1)
pp.pprint({
    'list multiply': list1 * 6,
    'np array multiply scalar': a * 6,
    'tensor multiply scalar': t * 6,
})


# numpy matmul
c = np.matmul(n, b)
pp.pprint(c)
pp.pprint(c.shape)

# torch matmul
a = torch.ones((4, 3)) * 6
b = torch.ones(3) * 2
c = b.T
d = a @ c
pp.pprint({
    'a.shape': a.shape,
    'tensor multiply scalar': b,
    'tensor multiply scalar shape': b.shape,
    'transpose of single dimension matrix returns same matrix': c,
    'transpose of single dimension matrix shape': c.shape,
    'matrix multiply': d,
    'matrix multiply shape': d.shape,
})

b = torch.ones(3, 1) * 2
c = b.T
d = a @ b
pp.pprint({
    'tensor multiply scalar': b,
    'tensor multiply scalar shape': b.shape,
    'transpose of two dim matrix': c,
    'c.shape': c.shape,
    'matrix multiply': d,
    'matrix multiply shape': d.shape,
})

# matmul applies mat multiplication at the lowest dimension
# and repeating across higher dimensions
a = torch.ones((5, 4, 3)) * 6
b = torch.ones(3) * 2
c = a @ b.T
pp.pprint(c)
pp.pprint(c.shape)

a = torch.ones((5, 4, 3)) * 6
b = torch.ones(3, 2) * 2
c = a @ b
pp.pprint(c)
pp.pprint(c.shape)


# the propagation of matrix operation (add, multiply etc) at lower dimension, to higher dimension
# In other words, repeating the operation at higher dimension
# comes from numpy.  Example below with add operation for 5 x 4 x 3 matrix with 4 x 3 matrix
# The add operation between the 4 x 3 matrices is propagated to the third dimension
n1 = np.ones((5, 4, 3))
n2 = np.array([[2, 3, 4],
              [3, 4, 5],
              [4, 5, 6],
              [5, 6, 7]])
pp.pprint(n1 + n2)