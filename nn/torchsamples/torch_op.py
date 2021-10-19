import numpy as np
import torch

import pprint
pp = pprint.PrettyPrinter()
np.set_printoptions(precision=4, suppress=True)

t = torch.randint(0, 10, (4, 3))

i = t.index_select(0, torch.tensor([1, 3]))

pp.pprint({
    't': t,
    'index_select': i
})

t1 = torch.tensor([[1, 2, 3],
                  [0, 1, 2]])
index_select_flatten = t.index_select(0, t1.flatten())

# using -1 to infer size
unflatten = index_select_flatten.unflatten(0, (2, -1))

unflatten_1 = index_select_flatten.unflatten(0, t1.shape)

pp.pprint({
    'index_select_flatten': index_select_flatten,
    'unflatten': unflatten,
    'unflatten_shape': unflatten_1,
})

# view to keen number of dimensions the same, but different dimension size
v = index_select_flatten.view(t1.shape[0], -1)
pp.pprint({
    'view': v
})

