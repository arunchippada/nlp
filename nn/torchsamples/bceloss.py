import torch
import torch.nn as nn

import pprint
pp = pprint.PrettyPrinter()

bceloss = nn.BCELoss()

output = torch.empty(3).random_(2)
target = torch.empty(3).random_(2)
loss = bceloss.forward(output, target)
pp.pprint({
    'output': output,
    'target': target,
    'loss': loss
})

output = torch.Tensor([
    [1, 0],
    [1, 0]
])

target = torch.Tensor([
    [1, 1],
    [1, 1]
])

loss = bceloss.forward(output, target)

bceloss_nr = nn.BCELoss(reduction='none')
loss_nr = bceloss_nr.forward(output, target)

pp.pprint({
    'output': output,
    'target': target,
    'loss': loss,
    'loss Noreduce': loss_nr
})