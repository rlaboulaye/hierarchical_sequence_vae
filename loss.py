import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss

class SequenceVariationalLoss(nn.Module):

	def __init__(self):
		super(SequenceVariationalLoss, self).__init__()
		self.reconstruction_loss = CrossEntropyLoss()

	def kld_loss(self, mu, logvar):
		return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	def forward(self, sequence_of_logits, sequence_of_targets, mu, logvar):
		losses = []
		for logits, targets in zip(sequence_of_logits, sequence_of_targets):
			batch_reconstruction_loss = self.reconstruction_loss(logits, targets)
			batch_kld_loss = self.kld_loss(mu, logvar)
			batch_loss = batch_reconstruction_loss + batch_kld_loss
			losses.append(batch_loss)
		return np.mean(losses)
