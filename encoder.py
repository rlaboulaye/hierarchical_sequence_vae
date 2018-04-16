import numpy as np
import torch
from torch import nn

from variable import get_variable
from rnn import RNN

from initialization import initialize_weights


class Encoder(nn.Module):

	def __init__(self, input_dimension=300, hidden_dimension=512, num_layers=3):
		super(Encoder, self).__init__()
		self.input_dimension = input_dimension
		self.hidden_dimension = hidden_dimension
		self.num_layers = num_layers
		self.forward_rnn = RNN(self.input_dimension, self.hidden_dimension, self.num_layers)
		self.backward_rnn = RNN(self.input_dimension, self.hidden_dimension, self.num_layers)
		self.mean_extractor = nn.Linear(self.hidden_dimension * 2, self.hidden_dimension)
		self.logvar_extractor = nn.Linear(self.hidden_dimension * 2, self.hidden_dimension)
		self.step_count = 0
		self.example_count = 0
		self.initialize_modules()

	def initialize_modules(self):
		for module in self.modules():
			module.apply(initialize_weights)

	def forward(self, input_sequence, batch_size=16):
		h_0 = get_variable(torch.FloatTensor(np.zeros((self.num_layers, batch_size, self.hidden_dimension))))
		forward_h_tm1 = h_0
		backward_h_tm1 = h_0
		sequence_length = len(input_sequence)
		embeddings = [[None, None]] * sequence_length
		for i in range(sequence_length):
			forward_input_embedding = input_sequence[i]
			backward_input_embedding = input_sequence[sequence_length - i - 1]
			forward_h_tm1 = self.forward_rnn(forward_input_embedding, forward_h_tm1)
			backward_h_tm1 = self.backward_rnn(backward_input_embedding, backward_h_tm1)
		sequence_embedding = torch.cat((forward_h_tm1[-1], backward_h_tm1[-1]), dim=-1)
		mu = self.mean_extractor(sequence_embedding)
		logvar = self.logvar_extractor(sequence_embedding)
		return mu, logvar

	def increment_step(self, step_count=1, batch_size=16):
		self.step_count += step_count
		self.example_count += step_count * batch_size
