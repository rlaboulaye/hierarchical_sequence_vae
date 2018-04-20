import math

import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss

from variable import get_variable

class SequenceVariationalLoss(nn.Module):

	def __init__(self):
		super(SequenceVariationalLoss, self).__init__()
		self.reconstruction_coefficient = 1.
		self.reconstruction_loss = CrossEntropyLoss()

	def _get_kld_coefficient(self, i):
		return (math.tanh((i - 166250) / 40000) + 1) / 2

	def kld_loss(self, mu, logvar, clip_value = 2.):
		loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)).mean()
		clip_mask = (loss < clip_value).float()
		keep_mask = (loss >= clip_value).float()
		clip_values = get_variable(torch.FloatTensor([clip_value]))
		loss = keep_mask * loss + clip_mask * clip_values
		return loss

	def forward(self, sequence_of_logits, sequence_of_targets, mu, logvar, iteration):
		reconstruction_losses = []
		for logits, targets in zip(sequence_of_logits, sequence_of_targets):
			batch_reconstruction_loss = self.reconstruction_loss(logits, targets)
			reconstruction_losses.append(batch_reconstruction_loss)
		kld_loss = self.kld_loss(mu, logvar)
		reconstruction_loss = torch.cat(reconstruction_losses).mean()
		loss = (self.reconstruction_coefficient * reconstruction_loss) + \
				(self._get_kld_coefficient(iteration) * kld_loss)
		return loss, reconstruction_loss, kld_loss
