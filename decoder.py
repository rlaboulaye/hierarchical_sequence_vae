import numpy as np

import torch
from torch import nn

from initialization import initialize_weights
from variable import get_variable
from rnn import RNN
from context_enhanced_rnn import ContextEnhancedRNN

class Decoder(nn.Module):
        
    def __init__(self, input_dimension=300, output_dimension=1000, hidden_dimension=512, \
            num_layers=3, context_dimension=None):
        super(Decoder, self).__init__()
        self.max_sequence_length = 20
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.context_dimension = context_dimension
        self.hidden_dimension = hidden_dimension
        self.num_layers = num_layers
        self.step_count = 0
        self.example_count = 0
        self.fc = nn.Linear(self.hidden_dimension, self.output_dimension)
        self.generating_activation = nn.Softmax(dim=1)
        if self.context_dimension is None:
            self.rnn = RNN(self.input_dimension, self.hidden_dimension, self.num_layers)
        else:
            self.rnn = ContextEnhancedRNN(self.input_dimension, self.hidden_dimension, \
                    self.context_dimension, self.num_layers)
        self.initialize_modules()

    def initialize_modules(self):
        for module in self.modules():
            module.apply(initialize_weights)

    def forward(self, context, embedding_dict, inputs=None, training_sequence_length=None, drop_prob=None, \
            batch_size=16, eos_token='.', unk_token='<unknown>'):
        eos_index = embedding_dict.get_index(eos_token)
        if drop_prob is not None:
            unk_index = embedding_dict.get_index(unk_token)
            bernoulli = torch.distributions.Bernoulli(torch.FloatTensor([drop_prob]))
        hidden_tm1 = context.repeat(self.num_layers, 1).view(self.num_layers, batch_size, -1)
        input_t = get_variable(torch.FloatTensor([embedding_dict[eos_token]] * batch_size))
        word_indices = [-1] * batch_size
        sequence_of_indices = []
        sequence_of_logits = []
        while (training_sequence_length is None and np.any(np.array(word_indices) != eos_index) \
                and len(sequence_of_indices) < self.max_sequence_length) or \
                (training_sequence_length is not None and len(sequence_of_indices) < training_sequence_length):
            if self.context_dimension is None:
                hidden_t = self.rnn(input_t, hidden_tm1)
            else:
                hidden_t = self.rnn(input_t, hidden_tm1, context)
            logits = self.fc(hidden_t[-1])
            probabilities = self.generating_activation(logits)
            word_indices = torch.multinomial(probabilities, 1).view(-1)
            sequence_of_logits.append(logits)
            sequence_of_indices.append(word_indices)
            if inputs is None:
                word_indices = word_indices.cpu().data
            else:
                word_indices = inputs[len(sequence_of_indices) - 1].cpu().data
                if drop_prob is not None:
                    drop_mask = bernoulli.sample_n(batch_size).view(-1).byte()
                    word_indices[drop_mask] = torch.LongTensor([unk_index] * drop_mask.sum())
            input_t = get_variable(torch.FloatTensor([embedding_dict[embedding_dict.get_word(word_index)] for word_index in word_indices.numpy()]))
            hidden_tm1 = hidden_t
        return sequence_of_logits, sequence_of_indices

    def increment_step(self, step_count=1, batch_size=16):
        self.step_count += step_count
        self.example_count += step_count * batch_size
