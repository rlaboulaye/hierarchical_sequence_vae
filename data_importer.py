import io
import math
import time
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

class BookSentences(Dataset):

    def __init__(self, min_length = 1, max_length = 20):
        self.max_length = max_length
        self.min_length = min_length
        self.data = []
        self.token_counts = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return BookSentences.parse(self.data[idx])

    @staticmethod
    def parse(sentence):
        result = []
        for token in sentence.strip().split():
            if not token == u'.':
                result.append(token)
        result.append(u'.')
        return result

    def append(self, sentence):
        self.data.append(sentence)

    # 74004229 rows, 8 GB memory, ~60s to load entire file with no embeddings
    # 67334174 rows, 8 GB memory, ~430s to load entire file with glove embeddings
    @staticmethod
    def load_from_file(file_name = "books_in_sentences.txt", max_rows = 1e5, max_length = None):
        book_sentences = BookSentences()
        count = 0
        with io.open(file_name, 'r', encoding = 'utf8') as data_file:
            for sentence in data_file:
                book_sentences.append(sentence)
                count += 1
                if max_rows is not None and count >= max_rows:
                    break
        return book_sentences

    @staticmethod
    def load_by_length(sentence_file = "books_in_sentences.txt", token_file = "most_common_tokens.txt", \
                       max_rows = 1e5, min_length = 1, max_length = 20, max_rarity = None):
        start_time = time.time()
        data = [BookSentences(min_length=x, max_length=x) for x in range(min_length, max_length + 1)]

        token_index = {}
        token_count = 0
        with open(token_file, 'r') as sf:
            for token in sf:
                token_index[token.strip()] = token_count
                token_count += 1

        count = 0
        with io.open(sentence_file, 'r', encoding = 'utf8') as data_file:
            for sentence in data_file:
                parsed_sentence = BookSentences.parse(sentence)
                length = len(parsed_sentence)
                rarity = max([token_index[t] for t in parsed_sentence])

                if max_length is not None and length > max_length:
                    continue

                if min_length is not None and length < min_length:
                    continue

                if max_rarity is not None and rarity > max_rarity:
                    continue

                data[length - min_length].append(' '.join(parsed_sentence))
                count += 1

                if max_rows is not None and count >= max_rows:
                    break

        print("Loaded "+ str(len(data)) +" datasets in {0:.2f} seconds".format(time.time() - start_time))
        return data


    @staticmethod
    def load_most_common_tokens(load_file = "books_in_sentences.txt", save_file = "most_common_tokens.txt", max_vocab_size = 1e3):
        start_time = time.time()
        try:
            with io.open(save_file, 'r', encoding = 'utf8') as sf:
                common_tokens = []
                for line in sf:
                    common_tokens.append(line.strip())
                if len(common_tokens) == 0:
                    raise ValueError("Save file doesn't exist")
        except:
            token_counts = {}
            with io.open(load_file, 'r', encoding = 'utf8') as lf:
                for sentence in lf:
                    for token in sentence.strip().split():
                        if token not in token_counts:
                            token_counts[token] = 1
                        else:
                            token_counts[token] += 1
            token_order = sorted([(token_counts[token], token) for token in token_counts], reverse=True)
            common_tokens = [x[1] for x in token_order]

            with io.open(save_file, 'w', encoding = 'utf8') as sf:
                for token_count, token in common_tokens:
                    sf.write(token + '\n')

        if max_vocab_size is None:
            print("Loaded "+ str(len(common_tokens)) +" tokens in {0:.2f} seconds".format(time.time() - start_time))
            return common_tokens
        else:
            print("Loaded "+ str(max_vocab_size) +" tokens in {0:.2f} seconds".format(time.time() - start_time))
            return common_tokens[:max_vocab_size]

class BookParagraphs(Dataset):

    def __init__(self, min_length = 1, max_length = 20):
        self.max_length = max_length
        self.min_length = min_length
        self.data = []
        self.token_counts = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def load_from_file(sentence_file = "books_in_sentences.txt", token_file = "most_common_tokens.txt", \
                       max_sentences=1e5, max_paragraphs=1e5, \
                       min_sentence_length=1, max_sentence_length=20, \
                       min_paragraph_length=1, max_paragraph_length=20, max_rarity = None):
        start_time = time.time()
        bp = BookParagraphs()

        token_index = {}
        token_count = 0
        with open(token_file, 'r') as sf:
            for token in sf:
                token_index[token.strip()] = token_count
                token_count += 1

        count = 0
        with io.open(sentence_file, 'r', encoding = 'utf8') as data_file:
            next_paragraph = BookSentences(min_length=min_sentence_length, max_length=max_sentence_length)

            for sentence in data_file:
                parsed_sentence = BookSentences.parse(sentence)
                length = len(parsed_sentence)
                rarity = max([token_index[t] for t in parsed_sentence])

                if (min_sentence_length is not None and length < min_sentence_length) \
                or (max_sentence_length is not None and length > max_sentence_length) \
                or (max_rarity is not None and rarity > max_rarity) \
                or (max_paragraph_length is not None and len(next_paragraph) >= max_paragraph_length):
                    if len(next_paragraph) >= min_paragraph_length:
                        count += len(next_paragraph)
                        bp.data.append(next_paragraph.data)
                        next_paragraph = BookSentences(min_length=min_sentence_length, max_length=max_sentence_length)
                    continue

                next_paragraph.append(parsed_sentence)

                if (max_sentences is not None and count >= max_sentences) \
                or (max_paragraphs is not None and len(bp) >= max_paragraphs):
                    break

        print("Loaded "+ str(len(bp)) +" paragraphs in {0:.2f} seconds".format(time.time() - start_time))
        return bp

# 2196016 rows, 6 GB memory, ~25 seconds to load entire file
class GloveEmbeddings():

    def __init__(self, file_name = "glove.txt", vocabulary = None):
        start_time = time.time()
        self.file_name = file_name
        self.data = {}
        self.index_to_token_map = {}
        self.count = 0
        self.vocabulary = vocabulary
        if self.vocabulary is not None:
            self.vocabulary = set(self.vocabulary)
            self.vocabulary.add(u".") # use period for end of sentence
            self.vocabulary.add(u"0") # use zero for numbers
            self.vocabulary.add(u"<unknown>") # use <unknown> for oov words

        with io.open(file_name, 'r', encoding = 'utf8') as data_file:
            for line in data_file:
                line = line.split(" ", 1)
                self.add(line[0], line[1])
        self.add(u"<unknown>", " ".join(["0"] * len(line[1].split(" "))))
        print("Loaded "+ str(len(self.data)) +" embeddings in {0:.2f} seconds".format(time.time() - start_time))


    def add(self, word, embedding):
        if self.vocabulary is not None:
            if word not in self.vocabulary:
                return False
        self.data[word] = (embedding, self.count)
        self.index_to_token_map[self.count] = word
        self.count += 1
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return [float(x) for x in self.data[key][0].split(" ")]

    def get_word(self, index):
        return self.index_to_token_map[index]

    def get_index(self, token):
        if token.isnumeric():
            return self.data[u"0"][1]
        elif token not in self.data:
            return self.data[u"<unknown>"][1]
        else:
            return self.data[token][1]

    def get_indexes(self, tokens):
        indexed_tokens = []
        for token in tokens:
            indexed_tokens.append(self.get_index(token))
        return indexed_tokens

    def __contains__(self, key):
        return key in self.data

    def embed_batch(self, batch):
        embedded_batch = []
        for token in batch:
            word = token
            if token.isnumeric():
                word = u"0"
            elif token not in self.data:
                word = u"<unknown>"
            embedded_batch.append(self[word])
        return embedded_batch

    def index_batch(self, batch):
        indexed_batch = []
        for token in batch:
            word = token
            if token.isnumeric():
                word = u"0"
            elif token not in self.data:
                word = u"<unknown>"
            indexed_batch.append(self.get_index(word))
        return indexed_batch

    def save(self, save_file = None):
        if save_file is None:
            save_file = "glove_" + str(len(self.data)) + ".txt"
        with io.open(save_file, 'w', encoding = 'utf8') as sf:
            for token in self.data:
                sf.write(token + " " + self.data[token][0])



class GloveDataset(Dataset):
    def __init__(self, file_name = "glove.txt", max_rows = 1e5):
        self.max_rows = max_rows
        self.file_name = file_name
        self.data = []

        with io.open(file_name, 'r', encoding = 'utf8') as data_file:
            for line in data_file:
                self.data.append(line)
                if self.max_rows is not None and len(self.data) >= self.max_rows:
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].split(" ")
        line = [line[0]] + [float(x) for x in line[1:]]
        return line

class CharacterEmbeddings():

    def __init__(self, file_name = "character_embedding_weights.txt"):
        with io.open(file_name, 'r') as f:
            character_to_index = {}
            index_to_character = {}
            entries = f.read().split()
            embedding_size  =300
            row_size = embedding_size + 1
            embedding = None
            for i in range(math.ceil(len(entries) / float(row_size))):
                character = entries[i * row_size]
                row = np.array(entries[i * row_size + 1 : i * row_size + row_size], dtype=float).reshape(1,-1)
                character_to_index[character] = i
                index_to_character[i] = character
                if embedding is None:
                    embedding = row
                else:
                    embedding = np.append(embedding, row, axis=0)
            self.embedding = embedding
            self.character_to_index = character_to_index
            self.index_to_character = index_to_character

    def to_indices(self, string):
        if len(string) > 20:
            return []
        return [self.character_to_index[x] for x in string if x in self.character_to_index]
