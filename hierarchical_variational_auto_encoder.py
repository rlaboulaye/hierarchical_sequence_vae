import os
import time
import itertools
import re

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from rnn import RNN
from encoder import Encoder
from decoder import Decoder
from loss import SequenceVariationalLoss
from error_rate import ErrorRate
from variable import get_variable
from data_importer import BookSentences, GloveEmbeddings,BookParagraphs


class HierarchicalVariationalAutoEncoder(nn.Module):

    def __init__(
        self,
        vocab_size=1000,
        input_dimension=300,
        hidden_dimension=512,
        num_layers=2,
        drop_prob=None,
        use_context_enhanced_rnn=True,
        use_pretrained_weights=False,
        min_sentence_length=5,
        max_sentence_length=9,
        min_paragraph_length=3,
        max_paragraph_length=3,
        max_rows=None,
        max_sentences_in_paragraph_loading=None,
        max_paragraphs=None
    ):
        super(HierarchicalVariationalAutoEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.input_dimension = input_dimension
        self.encoder_hidden_dimension = hidden_dimension
        self.decoder_hidden_dimension = hidden_dimension
        self.guide_hidden_dimension = hidden_dimension * 2
        self.num_layers = num_layers
        self.drop_prob = drop_prob
        self.use_pretrained_weights = use_pretrained_weights
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.min_paragraph_length = min_paragraph_length
        self.max_paragraph_length = max_paragraph_length
        self.identifier = '{}tokens_{}smin_{}smax_{}pmin_{}pmax_{}hidden_{}layers_{}drop_{}'.format(self.vocab_size, \
                self.min_sentence_length, self.max_sentence_length, self.min_paragraph_length, self.max_paragraph_length, \
                self.encoder_hidden_dimension, self.num_layers, 'no' if self.drop_prob is None else round(self.drop_prob * 100), \
                'contextenhancedrnn' if use_context_enhanced_rnn else 'simplernn')
        self._init_paths()
        self._load_data(max_rows=max_rows, max_sentences_in_paragraph_loading=max_sentences_in_paragraph_loading, \
                max_paragraphs=max_paragraphs)
        self._init_encoder(use_pretrained_weights)
        self._init_decoder(use_pretrained_weights, use_context_enhanced_rnn)
        self._init_guide(use_pretrained_weights)
        self.loss = SequenceVariationalLoss()
        self.vae_error_rate = ErrorRate()
        self.guide_loss = nn.L1Loss()
        self._init_cuda()

    def _init_paths(self):
        self.encoder_weights = 'weights/' + self.identifier + '_encoder.weights'
        self.decoder_weights = 'weights/' + self.identifier + '_decoder.weights'
        self.guide_weights = 'weights/' + self.identifier + '_guide.weights'
        self.vae_train_loss_path = 'results/{}_vae_train_loss.npy'.format(self.identifier)
        self.vae_test_loss_path = 'results/{}_vae_test_loss.npy'.format(self.identifier)
        self.vae_train_error_path = 'results/{}_vae_train_error.npy'.format(self.identifier)
        self.vae_test_error_path = 'results/{}_vae_test_error.npy'.format(self.identifier)
        self.guide_train_loss_path = 'results/{}_guide_train_loss.npy'.format(self.identifier)
        self.guide_test_loss_path = 'results/{}_guide_test_loss.npy'.format(self.identifier)
        self.guide_train_error_path = 'results/{}_guide_train_error.npy'.format(self.identifier)
        self.guide_test_error_path = 'results/{}_guide_test_error.npy'.format(self.identifier)

    def _init_encoder(self, use_pretrained_weights):
        if use_pretrained_weights == True and os.path.exists(self.encoder_weights):
            if torch.cuda.is_available():
                self.encoder = torch.load(self.encoder_weights)
            else:
                self.encoder = torch.load(self.encoder_weights, map_location=lambda storage, loc: storage)
        else:
            self.encoder = Encoder(self.input_dimension, self.encoder_hidden_dimension, self.num_layers)

    def _init_decoder(self, use_pretrained_weights, use_context_enhanced_rnn):
        if use_pretrained_weights == True and os.path.exists(self.decoder_weights):
            if torch.cuda.is_available():
                self.decoder = torch.load(self.decoder_weights)
            else:
                self.decoder = torch.load(self.decoder_weights, map_location=lambda storage, loc: storage)
        else:
            context_dimension = self.decoder_hidden_dimension
            if not use_context_enhanced_rnn:
                context_dimension = None
            self.decoder = Decoder(self.input_dimension, len(self.embeddings), self.decoder_hidden_dimension, \
                    self.num_layers, context_dimension)

    def _init_guide(self, use_pretrained_weights):
        if use_pretrained_weights == True and os.path.exists(self.guide_weights):
            if torch.cuda.is_available():
                self.guide = torch.load(self.guide_weights)
            else:
                self.guide = torch.load(self.guide_weights, map_location=lambda storage, loc: storage)
        else:
            self.guide = RNN(self.encoder_hidden_dimension * 2, self.guide_hidden_dimension, self.num_layers)

    def _init_cuda(self):
        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.guide = self.guide.cuda()
            self.loss = self.loss.cuda()
            self.vae_error_rate = self.vae_error_rate.cuda()
            self.guide_loss = self.guide_loss.cuda()

    def _load_data(self, glove_file = 'glove_9902.txt', sentence_file = 'books_in_sentences.txt', \
            token_file = "most_common_tokens.txt", max_rows=None, \
            max_sentences_in_paragraph_loading=None, max_paragraphs=None):
        self.most_common_tokens = BookSentences.load_most_common_tokens(max_vocab_size = self.vocab_size)
        self.embeddings = GloveEmbeddings(glove_file, vocabulary=self.most_common_tokens)
        self.book_sentence_datasets = BookSentences.load_by_length(sentence_file = sentence_file, \
                token_file = token_file, min_length = self.min_sentence_length, \
                max_length = self.max_sentence_length, max_rows=max_rows, max_rarity=self.vocab_size)
        self.book_paragraph_dataset = BookParagraphs.load_from_file(sentence_file=sentence_file, \
            max_sentences=max_sentences_in_paragraph_loading, max_paragraphs=max_paragraphs, \
            min_sentence_length=self.min_sentence_length, max_sentence_length=self.max_sentence_length, \
            min_paragraph_length=self.min_paragraph_length, max_paragraph_length=self.max_paragraph_length)

    def _get_sentence_data_loaders(self, batch_size, test_split_ratio=0.1):
        test_loaders = []
        train_loaders = []
        for i, dataset in enumerate(self.book_sentence_datasets):
            train, test = self._get_loaders(batch_size, dataset)
            train_loaders.append(train)
            test_loaders.append(test)
        return train_loaders, test_loaders

    def _get_paragraph_data_loader(self, batch_size, test_split_ratio=0.1):
        train_loader, test_loader = self._get_loaders(batch_size, self.book_paragraph_dataset)
        return train_loader, test_loader

    def _get_loaders(self, batch_size, dataset, test_split_ratio=0.1):
        test_split = int(test_split_ratio*len(dataset))
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        test_sampler = SubsetRandomSampler(indices[:test_split])
        train_sampler = SubsetRandomSampler(indices[test_split:])
        test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=batch_size)
        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
        return train_loader, test_loader

    def _get_vae_history(self):
        if self.use_pretrained_weights and os.path.exists(self.vae_train_loss_path) and os.path.exists(self.vae_train_error_path):
            train_losses = np.load(self.vae_train_loss_path).tolist()
            train_error_rates = np.load(self.vae_train_error_path).tolist()
        else:
            train_losses = [[],[],[]]
            train_error_rates = []

        if self.use_pretrained_weights and os.path.exists(self.vae_test_loss_path) and os.path.exists(self.vae_test_error_path):
            test_losses = np.load(self.vae_test_loss_path).tolist()
            test_error_rates = np.load(self.vae_test_error_path).tolist()
        else:
            test_losses = [[],[],[]]
            test_error_rates = []
        return train_losses, test_losses, train_error_rates, test_error_rates

    def _get_guide_history(self):
        if self.use_pretrained_weights and os.path.exists(self.guide_train_loss_path) and os.path.exists(self.guide_train_error_path):
            train_losses = np.load(self.guide_train_loss_path).tolist()
            train_error_rates = np.load(self.guide_train_error_path).tolist()
        else:
            train_losses = [[],[],[]]
            train_error_rates = []

        if self.use_pretrained_weights and os.path.exists(self.guide_test_loss_path) and os.path.exists(self.guide_test_error_path):
            test_losses = np.load(self.guide_test_loss_path).tolist()
            test_error_rates = np.load(self.guide_test_error_path).tolist()
        else:
            test_losses = [[],[],[]]
            test_error_rates = []
        return train_losses, test_losses, train_error_rates, test_error_rates

    def train_guide(self, num_epochs=40, train_epoch_size=950, test_epoch_size=50, learning_rate=1e-5, batch_size=5):
        optimizer = torch.optim.Adam(self.guide.parameters(), lr=learning_rate)
        train_losses, test_losses, train_error_rates, test_error_rates = self._get_guide_history()
        test_split_ratio = test_epoch_size / float(train_epoch_size + test_epoch_size)
        train_loader, test_loader = self._get_paragraph_data_loader(1, test_split_ratio)
        print('Train GUIDE')
        start_time = time.time()
        for e in range(num_epochs):
            print('Epoch {}'.format(e))
            print('Train')
            train_loss, train_r_loss, train_kld_loss, train_error_rate = self._guide_epoch(train_loader, train_epoch_size * batch_size, batch_size, optimizer)
            train_losses[0] += train_loss
            train_losses[1] += train_r_loss
            train_losses[2] += train_kld_loss
            train_error_rates += train_error_rate
            torch.save(self.guide, self.guide_weights)
            np.save(self.guide_train_loss_path, np.array(train_losses))
            np.save(self.guide_train_error_path, np.array(train_error_rates))
            if test_epoch_size > 0:
                print('Test')
                test_loss, test_r_loss, test_kld_loss, test_error_rate = self._guide_epoch(test_loader, test_epoch_size * batch_size, batch_size, None)
                test_losses[0] += test_loss
                test_losses[1] += test_r_loss
                test_losses[2] += test_kld_loss
                test_error_rates += test_error_rate
                np.save(self.guide_test_loss_path, np.array(test_losses))
                np.save(self.guide_test_error_path, np.array(test_error_rates))
            print('Elapsed Time: {}\n'.format(time.time() - start_time))

    def train_vae(self, num_epochs=70, train_epoch_size=4750, test_epoch_size=250, learning_rate=1e-5, batch_size=16):
        optimizer = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=learning_rate)
        train_losses, test_losses, train_error_rates, test_error_rates = self._get_vae_history()
        test_split_ratio = test_epoch_size / float(train_epoch_size + test_epoch_size)
        train_loaders, test_loaders = self._get_sentence_data_loaders(batch_size, test_split_ratio)
        train_lengths = np.array([len(data_loader) for data_loader in train_loaders], dtype=np.float32)
        test_lengths = np.array([len(data_loader) for data_loader in test_loaders], dtype=np.float32)
        print('Train VAE')
        start_time = time.time()
        for e in range(num_epochs):
            print('Epoch {}'.format(e))
            print('Train')
            sentence_length_indices = np.random.multinomial(1, \
                    .9999 * train_lengths / float(np.sum(train_lengths)), size=(train_epoch_size)).argmax(axis=1)
            train_loss, train_r_loss, train_kld_loss, train_error_rate = self._vae_epoch(train_loaders, sentence_length_indices, batch_size, optimizer)
            train_losses[0] += train_loss
            train_losses[1] += train_r_loss
            train_losses[2] += train_kld_loss
            train_error_rates += train_error_rate
            torch.save(self.encoder, self.encoder_weights)
            torch.save(self.decoder, self.decoder_weights)
            np.save(self.vae_train_loss_path, np.array(train_losses))
            np.save(self.vae_train_error_path, np.array(train_error_rates))
            if test_epoch_size > 0:
                print('Test')
                sentence_length_indices = np.random.multinomial(1, \
                        .9999 * test_lengths / float(np.sum(test_lengths)), size=(test_epoch_size)).argmax(axis=1)
                test_loss, test_r_loss, test_kld_loss, test_error_rate = self._vae_epoch(test_loaders, sentence_length_indices, batch_size, None)
                test_losses[0] += test_loss
                test_losses[1] += test_r_loss
                test_losses[2] += test_kld_loss
                test_error_rates += test_error_rate
                np.save(self.vae_test_loss_path, np.array(test_losses))
                np.save(self.vae_test_error_path, np.array(test_error_rates))
            print('Elapsed Time: {}\n'.format(time.time() - start_time))

    def _vae_epoch(self, loaders, sentence_length_indices, batch_size, optimizer=None):
        losses = []
        reconstruction_losses = []
        kld_losses = []
        error_rates = []
        for index in sentence_length_indices:
            loader = loaders[index]
            sequence = next(iter(loader))
            sequence_of_embedded_batches = [get_variable(torch.FloatTensor(self.embeddings.embed_batch(batch))) for batch in sequence]
            sequence_of_indexed_batches = [get_variable(torch.LongTensor(self.embeddings.index_batch(batch))) for batch in sequence]
            mu, logvar = self._encoder_forward(sequence_of_embedded_batches, batch_size)
            context = self._get_context(mu, logvar, batch_size)

            if optimizer is not None:
                logits, predictions = self._decoder_forward(context, batch_size, sequence_of_indexed_batches, len(sequence), self.drop_prob)
            else:
                logits, predictions = self._decoder_forward(context, batch_size, None, len(sequence), None)

            loss, reconstruction_loss, kld_loss = self.loss(logits, sequence_of_indexed_batches, mu, logvar, self.decoder.step_count)
            losses.append(loss.cpu().data.numpy())
            reconstruction_losses.append(reconstruction_loss.cpu().data.numpy())
            kld_losses.append(kld_loss.cpu().data.numpy())

            error_rate = self.vae_error_rate(predictions, sequence_of_indexed_batches)
            error_rates.append(error_rate.cpu().data.numpy())

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.encoder.increment_step(batch_size=batch_size)
                self.decoder.increment_step(batch_size=batch_size)

        print('Mean Loss: {}'.format(np.mean(losses)))
        print('Mean Error Rate: {}'.format(np.mean(error_rates)))
        return losses, reconstruction_losses, kld_losses, error_rates

    def _guide_epoch(self, loader, num_iterations, batch_size, optimizer=None):
        losses = []
        reconstruction_losses = []
        kld_losses = []
        error_rates = []
        tmp_losses = []
        tmp_reconstruction_losses = []
        tmp_kld_losses = []
        tmp_error_rates = []
        for index in range(num_iterations):
            sequences = next(iter(loader))
            sequence = sequences[0]
            sequence_of_embedded_batches = [get_variable(torch.FloatTensor(self.embeddings.embed_batch(batch))) for batch in sequence]
            mu, logvar = self._encoder_forward(sequence_of_embedded_batches, 1)
            h_tm1 = get_variable(torch.zeros(self.num_layers, 1, self.guide_hidden_dimension))
            for sequence_i in range(1, len(sequences)):
                h_t = self.guide(torch.cat([mu, logvar], dim=1), h_tm1)
                mu, logvar = h_t[-1].split(self.decoder_hidden_dimension, dim=1)
                context = self._get_context(mu, logvar, 1)
                sequence = sequences[sequence_i]
                sequence_of_embedded_batches = [get_variable(torch.FloatTensor(self.embeddings.embed_batch(batch))) for batch in sequence]
                sequence_of_indexed_batches = [get_variable(torch.LongTensor(self.embeddings.index_batch(batch))) for batch in sequence]

                if optimizer is not None:
                    logits, predictions = self._decoder_forward(context, 1, sequence_of_indexed_batches, len(sequence), self.drop_prob)
                else:
                    logits, predictions = self._decoder_forward(context, 1, None, len(sequence), None)

                loss, reconstruction_loss, kld_loss = self.loss(logits, sequence_of_indexed_batches, mu, logvar, self.decoder.step_count)

                tmp_losses.append(loss)
                tmp_reconstruction_losses.append(reconstruction_loss)
                tmp_kld_losses.append(kld_loss)

                error_rate = self.vae_error_rate(predictions, sequence_of_indexed_batches)
                tmp_error_rates.append(error_rate)

                mu, logvar = self._encoder_forward(sequence_of_embedded_batches, 1)
                h_tm1 = h_t

            if (index + 1) % batch_size == 0:
                loss = torch.cat(tmp_losses).mean()
                reconstruction_loss = torch.cat(tmp_reconstruction_losses).mean()
                kld_loss = torch.cat(tmp_kld_losses).mean()
                error_rate = torch.cat(tmp_error_rates).mean()
                tmp_losses = []
                tmp_reconstruction_losses = []
                tmp_kld_losses = []
                tmp_error_rates = []
                losses.append(loss.cpu().data.numpy())
                reconstruction_losses.append(reconstruction_loss.cpu().data.numpy())
                kld_losses.append(kld_loss.cpu().data.numpy())
                error_rates.append(error_rate.cpu().data.numpy())
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        print('Mean Loss: {}'.format(np.mean(losses)))
        print('Mean Error Rate: {}'.format(np.mean(error_rates)))
        return losses, reconstruction_losses, kld_losses, error_rates

    def _encoder_forward(self, sequence_of_embedded_batches, batch_size):
        return self.encoder(sequence_of_embedded_batches, batch_size)

    def _get_context(self, mu, logvar, batch_size):
        z = get_variable(torch.randn(batch_size, self.decoder_hidden_dimension))
        std = torch.exp(0.5 * logvar)
        context = z * std + mu
        return context

    def _decoder_forward(self, context, batch_size, targets=None, sequence_length=None, drop_prob=None):
        logits, predictions = self.decoder(context, self.embeddings, targets, \
                sequence_length, drop_prob, batch_size)
        return logits, predictions

    def generate_sentence(self, batch_size=16):
        context = get_variable(torch.randn(batch_size, self.decoder_hidden_dimension))
        logits, predictions = self.decoder(context, self.embeddings, batch_size=batch_size)
        return self._format_sentences(self._to_sentences(predictions, batch_size))

    def interpolate(self, start_sentence, end_sentence, steps=5):
        start_split_sentence = start_sentence.split(" ")
        start_sequence_of_batches = [[word] for word in start_split_sentence]
        start_sequence_of_embedded_batches = [get_variable(torch.FloatTensor(self.embeddings.embed_batch(batch))) for batch in start_sequence_of_batches]
        start_mu, start_logvar = self._encoder_forward(start_sequence_of_embedded_batches, 1)

        end_split_sentence = end_sentence.split(" ")
        end_sequence_of_batches = [[word] for word in end_split_sentence]
        end_sequence_of_embedded_batches = [get_variable(torch.FloatTensor(self.embeddings.embed_batch(batch))) for batch in end_sequence_of_batches]
        end_mu, end_logvar = self._encoder_forward(end_sequence_of_embedded_batches, 1)

        step_size = (end_mu - start_mu) / float(steps)
        sentences = [start_sentence]
        for i in range(steps - 1):
            logits, predictions = self.decoder(start_mu + i * step_size, self.embeddings, batch_size=1)
            sentences.extend(self._to_sentences(predictions, 1))
        sentences.append(end_sentence)
        return self._format_sentences(sentences)

    def reconstruct(self, sentence):
        split_sentence = sentence.split(" ")
        sequence_of_batches = [[word] for word in split_sentence]
        sequence_of_embedded_batches = [get_variable(torch.FloatTensor(self.embeddings.embed_batch(batch))) for batch in sequence_of_batches]
        mu, logvar = self._encoder_forward(sequence_of_embedded_batches, 1)
        contexts = self._get_context(mu, logvar, 3)
        logits, mean_predictions = self._decoder_forward(mu, 1)
        logits, sample_predictions = self._decoder_forward(contexts, 3)
        return (self._format_sentences([sentence]),
            self._format_sentences(self._to_sentences(mean_predictions, 1)),
            self._format_sentences(self._to_sentences(sample_predictions, 3))
            )

    def _to_sentences(self, predictions, batch_size):
        sentences = [[] for i in range(batch_size)]
        for batch in predictions:
            np_batch = batch.cpu().data.numpy().reshape(-1)
            for i in range(len(np_batch)):
                sentences[i].append(self.embeddings.get_word(np_batch[i]))
        sentences = [sentence[:-1] if sentence[-2] in set(['!','?']) else sentence for sentence in sentences]
        sentences = [' '.join(sentence).split('.', 1)[0] + '.' for sentence in sentences]
        return sentences

    def _format_sentences(self, sentences):
        sentences = [re.sub(r' (\.)*(?P<capture>([a-z]*\'[a-z]+)|[,;:\.\\?\!"]|(\'\'))', r'\g<capture>', sentence.replace('`` ', '``')) for sentence in sentences]
        return sentences
