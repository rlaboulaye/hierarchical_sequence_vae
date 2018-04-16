import numpy as np
import torch
from torch import nn

class ErrorRate(nn.Module):

	def __init__(self):
		super(ErrorRate, self).__init__()

	def forward(self, sequence_of_predictions, sequence_of_targets):
		error_rates = []
		for predictions, targets in zip(sequence_of_predictions, sequence_of_targets):
			errors = (predictions != targets)
			error_rates.append(errors.mean())
		return np.mean(error_rates)
