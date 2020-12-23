# -*- coding: utf-8 -*-
""" Predictive models for text classification. """

from numpy import empty, argmax
from torch import nn


class CharCNN(nn.Module):
    """

    A character-level CNN for text classification.
    This architecture is an implementation of Zhang et al., 'Character-level Convolutional Networks for Text Classification', NeurIPS, 2016

    """

    def __init__(self,
                 alphabet_size,
                 max_seq_length,
                 num_classes,
                 num_conv_filters=256,
                 num_fc_filters=1024):

        super(CharCNN, self).__init__()

        self.__name__ = 'CharCNN'

        self.alphabet_size = alphabet_size
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes

        self.num_conv_filters = num_conv_filters
        self.num_fc_filters = num_fc_filters
        self.conv_kernel_sizes = [7, 7, 3, 3, 3, 3]
        self.pool_kernel_sizes = [3, 3, None, None, None, 3]

        # Calculate output length of last conv. layer
        self.conv_seq_length = self._calculate_conv_seq_length()

        # Define convolutional layers
        self.conv1 = nn.Sequential(nn.Conv1d(self.alphabet_size, num_conv_filters,
                                             kernel_size=7, padding=0),
                                   nn.ReLU(),
                                   nn.MaxPool1d(3))

        self.conv2 = nn.Sequential(nn.Conv1d(num_conv_filters, num_conv_filters,
                                             kernel_size=7, padding=0),
                                   nn.ReLU(),
                                   nn.MaxPool1d(3))

        self.conv3 = nn.Sequential(nn.Conv1d(num_conv_filters, num_conv_filters,
                                             kernel_size=3, padding=0),
                                   nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv1d(num_conv_filters, num_conv_filters,
                                             kernel_size=3, padding=0),
                                   nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv1d(num_conv_filters, num_conv_filters,
                                             kernel_size=3, padding=0),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Conv1d(num_conv_filters, num_conv_filters,
                                             kernel_size=3, padding=0),
                                   nn.ReLU(),
                                   nn.MaxPool1d(3))


        # Define fully-connected output layers
        self.fc1 = nn.Sequential(nn.Linear(self.conv_seq_length, num_fc_filters),
                                 nn.ReLU(),
                                 nn.Dropout(0.5))

        self.fc2 = nn.Sequential(nn.Linear(num_fc_filters, num_fc_filters),
                                 nn.ReLU(),
                                 nn.Dropout(0.5))

        self.fc_out = nn.Linear(num_fc_filters, self.num_classes)

        self._initialise_weights()

    def forward(self, x):
        """ Forward pass """

        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # Reshape
        x = x.view(x.size(0), -1)

        # Fully-connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc_out(x)

        return x

    def _calculate_conv_seq_length(self):
        """ Calculate number of units in output of last convolutional layer. """
        conv_seq_length = self.max_seq_length

        for fc, fp in zip(self.conv_kernel_sizes, self.pool_kernel_sizes):
            conv_seq_length = (conv_seq_length - fc) + 1

            if fp is not None:
                conv_seq_length = (conv_seq_length - fp)//fp + 1

        return conv_seq_length * self.num_conv_filters

    def _initialise_weights(self, mean=0.0, std=0.05):
        """ Initialise weights with Gaussian distribution. """
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)


class Baseline(object):
    """
    A simple baseline model that models each message as 1-grams of characters and learns marginal frequency counts from the training data
    """

    def __init__(self, alphabet, labels):
        self.alphabet = alphabet
        self.labels = labels
        self.frequency_counts = {}

    def train(self, df):
        groups = df.groupby('Label')

        for l in self.labels:
            messages = groups.get_group(l)['Txt']
            self.frequency_counts[l] = self._get_character_counts(messages)

    def predict(self, messages):
        assert self.frequency_counts is not None, 'Need to train model first'

        likelihood = empty((len(messages), len(self.labels)))

        for i, m in enumerate(messages):
            for l, freqs in self.frequency_counts.items():
                p = 1
                for char in m:
                    p *= freqs[char]

                likelihood[i, self.labels[l]] = p

        return likelihood

    def _get_character_counts(self, messages):
        counter = {char: 0 for char in self.alphabet}

        for m in messages:
            for char in self.alphabet:
                counter[char] += m.count(char)

        total = sum(counter.values())
        frequencies = {char: c /total for char, c in counter.items()}

        return frequencies

    def _get_labels(self, likelihood):

        return argmax(likelihood, axis=1)