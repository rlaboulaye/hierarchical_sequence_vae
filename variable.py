import torch
from torch import autograd

def get_variable(data, *args, **kwargs):
	variable = autograd.Variable(data, *args, **kwargs)
	if torch.cuda.is_available():
		variable = variable.cuda()
	return variable
