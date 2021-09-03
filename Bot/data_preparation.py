import json
import pickle
import numpy as np
import os

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


class PreTrainDataPrep:
    def __init__(self, intent_data, prep_data_filename="data.pickle"):
        self.prep_data_filename = prep_data_filename
        self.words = []
        self.labels = []
        self.training = []
        self.output = []
        self.intent_data = intent_data

    def load_preprocessed_data(self):
        if os.path.isfile(self.prep_data_filename):
            with open(self.prep_data_filename, "rb") as f:
                self.words, self.labels, self.training, self.output = pickle.load(f)
                return

        pattern_tokens = []
        intent_tags = []

        self.preprocess_words(pattern_tokens, intent_tags)
        self.one_hot_encode_training(pattern_tokens, intent_tags)

        with open(self.prep_data_filename, "wb") as f:
            pickle.dump((self.words, self.labels, self.training, self.output), f)

    def one_hot_encode_training(self, pattern_tokens, intent_tags):
        out_empty = [0] * len(self.labels)
        for x, doc in enumerate(pattern_tokens):
            stemmed_words = [stemmer.stem(word) for word in doc]
            bag = [1 if word in stemmed_words else 0 for word in self.words]

            output_row = out_empty[:]
            output_row[self.labels.index(intent_tags[x])] = 1

            self.training.append(bag)
            self.output.append(output_row)
        self.training = np.array(self.training)
        self.output = np.array(self.output)

    def preprocess_words(self, pattern_tokens, intent_tags):
        for intent in self.intent_data["intents"]:
            for pattern in intent["patterns"]:
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                pattern_tokens.append(word_list)
                intent_tags.append(intent["tag"])

            self.labels.append(intent["tag"])
        self.words = [stemmer.stem(word.lower()) for word in self.words if word != '?']
        self.words = sorted(list(set(self.words)))
        self.labels = sorted(self.labels)
