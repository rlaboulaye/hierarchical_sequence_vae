import torch
from torch import nn

from gru_cell import GRUCell


class RNN(nn.Module):

	def __init__(self, input_dimension, hidden_dimension, num_layers, time_cell=GRUCell):
		super(RNN, self).__init__()
		if num_layers < 1:
			raise ValueError('num_layers must be 1 or greater')
		self.input_dimension = input_dimension
		self.hidden_dimension = hidden_dimension
		self.layers = nn.ModuleList()
		for i in range(num_layers):
			if i == 0:
				self.layers.append(time_cell(self.input_dimension, self.hidden_dimension))
			else:
				self.layers.append(time_cell(self.hidden_dimension, self.hidden_dimension))

	def forward(self, x_t, h_tm1):
		h_t = []
		for i, layer in enumerate(self.layers):
			h_t.append(layer(x_t, h_tm1[i]))
			x_t = h_t[i]
		return h_t
