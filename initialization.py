import numpy as np
import torch
from torch import nn

def initialize_weights(m):
	if isinstance(m, nn.Linear):
		xavier_initialization(m)

def xavier_initialization(m):
	m.weight.data.normal_(0, np.sqrt(2. / (m.in_features + m.out_features)))
	m.bias.data.fill_(0.)
