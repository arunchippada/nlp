import numpy as np
import torch

from pprint import pprint, PrettyPrinter
pp = PrettyPrinter()

t = torch.randint(0, 10, (3, 5))
t1 = t[:-1]

pp.pprint({
    'original tensor': t,
    'slice in 0th dim': t1
})


