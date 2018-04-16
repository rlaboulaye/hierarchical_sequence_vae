import numpy as np
import torch
from torch import nn

from gru_cell import GRUCell


class ContextEnhancedGRUCell(GRUCell):

	def __init__(self, input_dimension, hidden_dimension, context_dimension, output_activation=nn.Tanh(), gating_activation=nn.Sigmoid()):
		self.context_dimension = context_dimension
		super(ContextEnhancedGRUCell, self).__init__(input_dimension + context_dimension, hidden_dimension, output_activation, gating_activation)

	def forward(self, x_t, h_tm1, c_t):
		x_t = torch.cat((x_t, c_t), dim=-1)
		z_t = self.gating_activation(self.W_z(x_t) + self.U_z(h_tm1))
		r_t = self.gating_activation(self.W_r(x_t) + self.U_r(h_tm1))
		h_t = (1 - z_t) * h_tm1 + z_t * self.output_activation(self.W_h(x_t) + self.U_h(r_t * h_tm1))
		return h_t
