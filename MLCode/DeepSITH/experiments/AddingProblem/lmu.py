"""https://github.com/AbdouJaouhar/LMU-Legendre-Memory-Unit"""

import torch
import torch.nn as nn
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
import numpy as np
from functools import partial
import torch.nn.functional as F
import math

from nengolib.signal import Identity, cont2discrete
from nengolib.synapses import LegendreDelay
from functools import partial

'''
Initialisation LECUN_UNIFOR
- tensor to fill
- fan_in is the input dimension size
'''
def lecun_uniform(tensor):
    fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
    nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))


class LMUCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                 order,
                 theta=100,  # relative to dt=1
                 method='zoh',
                 trainable_input_encoders=True,
                 trainable_hidden_encoders=True,
                 trainable_memory_encoders=True,
                 trainable_input_kernel=True,
                 trainable_hidden_kernel=True,
                 trainable_memory_kernel=True,
                 trainable_A=False,
                 trainable_B=False,
                 input_encoders_initializer=lecun_uniform,
                 hidden_encoders_initializer=lecun_uniform,
                 memory_encoders_initializer=partial(torch.nn.init.constant_, val=0),
                 input_kernel_initializer=torch.nn.init.xavier_normal_,
                 hidden_kernel_initializer=torch.nn.init.xavier_normal_,
                 memory_kernel_initializer=torch.nn.init.xavier_normal_,

                 hidden_activation='tanh',
                 ):
        super(LMUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.order = order

        if hidden_activation == 'tanh':
            self.hidden_activation = torch.tanh
        elif hidden_activation == 'relu':
            self.hidden_activation = torch.relu
        else:
            raise NotImplementedError("hidden activation '{}' is not implemented".format(hidden_activation))

        realizer = Identity()
        self._realizer_result = realizer(
            LegendreDelay(theta=theta, order=self.order))
        self._ss = cont2discrete(
            self._realizer_result.realization, dt=1., method=method)
        self._A = self._ss.A - np.eye(order)  # puts into form: x += Ax
        self._B = self._ss.B
        self._C = self._ss.C
        assert np.allclose(self._ss.D, 0)  # proper LTI

        self.input_encoders = nn.Parameter(torch.Tensor(1, input_size), requires_grad=trainable_input_encoders)
        self.hidden_encoders = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=trainable_hidden_encoders)
        self.memory_encoders = nn.Parameter(torch.Tensor(1, order), requires_grad=trainable_memory_encoders)
        self.input_kernel = nn.Parameter(torch.Tensor(hidden_size, input_size), requires_grad=trainable_input_kernel)
        self.hidden_kernel = nn.Parameter(torch.Tensor(hidden_size, hidden_size), requires_grad=trainable_hidden_kernel)
        self.memory_kernel = nn.Parameter(torch.Tensor(hidden_size, order), requires_grad=trainable_memory_kernel)
        self.AT = nn.Parameter(torch.Tensor(self._A), requires_grad=trainable_A)
        self.BT = nn.Parameter(torch.Tensor(self._B), requires_grad=trainable_B)

        # Initialize parameters
        input_encoders_initializer(self.input_encoders)
        hidden_encoders_initializer(self.hidden_encoders)
        memory_encoders_initializer(self.memory_encoders)
        input_kernel_initializer(self.input_kernel)
        hidden_kernel_initializer(self.hidden_kernel)
        memory_kernel_initializer(self.memory_kernel)

    def forward(self, input, hx):

        h, m = hx

        u = (F.linear(input, self.input_encoders) +
             F.linear(h, self.hidden_encoders) +
             F.linear(m, self.memory_encoders))

        m = m + F.linear(m, self.AT) + F.linear(u, self.BT)

        h = self.hidden_activation(
            F.linear(input, self.input_kernel) +
            F.linear(h, self.hidden_kernel) +
            F.linear(m, self.memory_kernel))

        return h, [h, m]


class LegendreMemoryUnit(nn.Module):
    """
    Implementation of LMU using LegendreMemoryUnitCell so it can be used as LSTM or GRU in PyTorch Implementation (no GPU acceleration)
    """
    def __init__(self, input_dim, hidden_size, order, theta):
        super(LegendreMemoryUnit, self).__init__()

        self.hidden_size = hidden_size
        self.order = order

        self.lmucell = LMUCell(input_dim, hidden_size, order, theta)

    def forward(self, xt):
        outputs = []

        h0 = torch.zeros(xt.size(0),self.hidden_size).cuda()
        m0 = torch.zeros(xt.size(0),self.order).cuda()
        states = (h0,m0)
        for i in range(xt.size(1)):
            out, states = self.lmucell(xt[:,i,:], states)
            outputs += [out]
        return torch.stack(outputs).permute(1,0,2), states

