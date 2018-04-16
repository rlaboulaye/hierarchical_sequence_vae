import numpy as np
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss

class SequenceVariationalLoss(nn.Module):

	def __init__(self, element_loss=CrossEntropyLoss):
		super(SequenceVariationalLoss, self).__init__()
		self.element_loss = element_loss(ignore_index=-1)

	def forward(self, sequence_of_logits, sequence_of_targets):
		losses = []
		for logits, targets in zip(sequence_of_logits, sequence_of_targets):
			batch_loss = self.element_loss(logits, targets)
			losses.append(batch_loss)
		return np.mean(losses)
