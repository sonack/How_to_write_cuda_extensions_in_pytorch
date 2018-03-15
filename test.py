import numpy as np
import unittest

import torch
from torch.autograd import Function, Variable
from round import round_forward_wrapper

class RoundFunction(Function):
    ''' test cuda pytorch
     round layer from caffe
    '''
    def forward(self, x):
        round_forward_wrapper(x, x, x.numel())
        return x

class TestRoundFunction(unittest.TestCase):
    def test_round_function(self):
        x = torch.rand(2,2).cuda()
        x_round = torch.zeros_like(x);
        x_round[x >= 0.5] = 1

        my_x_round = RoundFunction()(Variable(x))
        self.assertTrue(np.allclose(x_round.cpu().numpy(), my_x_round.data.cpu().numpy()))


if __name__ == '__main__':
    unittest.main()
