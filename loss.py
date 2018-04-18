import math

import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss

class SequenceVariationalLoss(nn.Module):

	def __init__(self):
		super(SequenceVariationalLoss, self).__init__()
		self.reconstruction_coefficient = 79
		self.reconstruction_loss = CrossEntropyLoss()

	def _get_kld_coefficient(self, iteration):
		return (math.tanh((iteration - 3500) / 1000) + 1) / 2

	def kld_loss(self, mu, logvar):
		return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)).mean()

	def forward(self, sequence_of_logits, sequence_of_targets, mu, logvar, iteration):
		reconstruction_losses = []
		for logits, targets in zip(sequence_of_logits, sequence_of_targets):
			batch_reconstruction_loss = self.reconstruction_loss(logits, targets)
			reconstruction_losses.append(batch_reconstruction_loss)
		kld_loss = self.kld_loss(mu, logvar)
		reconstruction_loss = torch.cat(reconstruction_losses).mean()
		loss = (self.reconstruction_coefficient * reconstruction_loss) + \
				(self._get_kld_coefficient(iteration) * kld_loss)
		return loss
