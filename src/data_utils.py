# -*- coding: utf-8 -*-
""" Helper functions for data manipulation and processing. """

import numpy as np
import torch

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from pandas import read_pickle, DataFrame

from src.cipher_utils import generate_key_list

from logging import getLogger
from logging.config import fileConfig

logconfig = Path('.') / 'logging_config.ini'
fileConfig(logconfig)
logger = getLogger()

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)

DELIMITER = chr(0x06)


def msg_to_cipher(messages, cipher_map, alphabet, key_rot=1000, key_map=None, label='cipher'):
    """
    Takes in a list of messages and applies the ciphers in cipher_map.

    :param messages: list:  List of messages with chars from alphabet
    :param cipher_map: dict: Dict of cipher objects
    :param alphabet: str: Alphabet containing all possible chars in messages
    :param key_rot: int: Frequency of key rotation. Will be ignored if key map is specified
    :param key_map: dict: Dict assigning a list of keys to each cipher
    :param label: str: Class label to be assigned to the ecnrypted messages

    :return: encrypted: dict: Dict containing for each cipher in cipher_map the list of encrypted messages, the list of keys for each message, the list of labels for each message
    """
    encrypted = {}

    for cipher in cipher_map:
        messages_en = []
        key_store = []

        cm = cipher_map[cipher]
        if cm is not None:
            print('--%s--'%cipher)
            cm.set_alphabet(alphabet)

            if key_map is not None:
                keys = key_map[cipher]
                key_rot = len(messages) // len(keys)
            else:
                nkeys = len(messages) // key_rot
                keys = generate_key_list(cipher, nkeys, alphabet)


            for i, k in enumerate(keys):
                cm.set_key(k)

                for m in messages[i * key_rot: (i+1) * key_rot]:
                    m_en = cm.encrypt(m)
                    messages_en.append(m_en)
                    key_store.append(k)

            print(f'Successfully encrypted all messages with cipher {cipher}')
            encrypted[cipher] = [messages_en, key_store, [label for _ in key_store]]

    return encrypted


def one_hot_encode(messages, char_to_token, max_seq_length):
    """
    One-hot encode a list of messages into a 0-1 array of size (num_chars x max_seq_length)

    :param messages: list: List of messages to encode
    :param char_to_token: dict: Mapping from char to index
    :param max_seq_length: int: Maximum length of each message

    :return: messages_one_hot: array: Binary array of one-hot encoded messages
    """
    num_char = len(char_to_token)
    messages_one_hot = np.zeros((len(messages), num_char, max_seq_length), dtype=np.float32)

    for i, m in enumerate(messages):
        for j, c in enumerate(str(m[:max_seq_length])[::-1]):
            try:
                messages_one_hot[i, char_to_token[c], j] = 1.
            except:
                pass # unknown characters will be encoded as all zeros

    return messages_one_hot


def build_tensor_dataset(data_path, config):
    """
    Read in (pkl) data object from given path, encode into numerical features and labels, and return a Tensor DataLoader object

    :param data_path: str:
    :param config:
    :return:
    """

    df = read_pickle(data_path)

    logger.info(f'Loaded dataset with {len(df)} samples')

    alphabet = get_ascii_alphabet()
    alphabet_dict = get_alphabet_dict(alphabet)
    logger.info(f'Loaded alphabet with {len(alphabet_dict)} characters')

    # If specified, drop all messages encrypted with drop_cipher
    if config['drop_cipher'] is not 'None':
        df = df[df['Cipher'] != config['drop_cipher']]

    # Extract binary feature matrix
    if config['max_num_samples'] is not None and len(df) > config['max_num_samples']:
        df = df.sample(config['max_num_samples'])

    messages = list(df['Txt'].values)
    features = one_hot_encode(messages, alphabet_dict, config['max_seq_length'])
    logger.info('Encoded messages as binary matrix')

    # Get labels
    labels = df['Label'].apply(lambda x: config["labels"][x]).values

    # Create Tensor dataset
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    loader = DataLoader(dataset, shuffle=True, batch_size=config['batch_size'])

    # Clean up
    del messages, labels, df

    return loader


def txt_to_list(fpath, delimiter=DELIMITER):
    """ Reads in messages from a txt file separated by the specified delimiter and stores them in a list. """
    with open(fpath, 'r') as f:
        content = f.read()
        messages = content.split(delimiter)
    print(f'Loaded {len(messages)} messages')

    tmp = 0
    for m in messages:
        if len(m) > tmp:
            tmp = len(m)
    print(f'Maximum message length: {tmp}')

    return messages


def generate_dataframe(plain_dict, cipher_dict):
    """ Takes two dictionaries with plain and ciphertext examples an combines them into a labelled DataFrame object. """
    txt = plain_dict['txt'].copy()
    labels = plain_dict['label'].copy()
    ciphers = ['plain' for _ in txt]
    keys = ['' for _ in txt]

    for cm in cipher_dict:
        messages_cm, keys_cm, labels_cm = cipher_dict[cm]
        n = len(messages_cm)

        txt.extend(messages_cm)
        keys.extend(keys_cm)
        labels.extend(labels_cm)
        ciphers.extend([cm for _ in range(n)])

    df = DataFrame({'Txt': txt, 'Label': labels, 'Cipher': ciphers, 'Key': keys})

    return df


def get_ascii_alphabet():
    """ Get alphabet for basic ASCI character set. excluding msg delimiter. """
    alphabet = ''.join([chr(i) for i in range(128) if chr(i) is not DELIMITER])

    return alphabet


def get_alphabet_dict(alphabet):
    """ Assign char to index mapping. """
    char_to_token = {c:i for i, c in enumerate(alphabet)}

    return char_to_token


def json_numpy_serialzer(o):
    """ Serialize numpy types for json. """
    numpy_types = (
        np.bool_,
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.str_,
        np.timedelta64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.void,
    )

    if isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, numpy_types):
        return o.item()
    elif isinstance(o, np.float128):
        return o.astype(np.float64).item()
    else:
        raise TypeError("{} of type {} is not JSON serializable".format(repr(o), type(o)))


def load_results(train_data_path, predictions_path):
    # Load predictions
    predictions = read_pickle(predictions_path)

    # Load training data
    train = read_pickle(train_data_path)

    train_cm_map = {}
    for cm, data in train.groupby('Cipher'):
        train_cm_map[cm] = list(data['Key'].unique())

    for cm in list(predictions['Cipher'].unique()):
        if cm not in train_cm_map:
            train_cm_map[cm] = []


    # Get prediction correct and whether key was in train data
    predictions['Prediction Correct'] = predictions['Label']== predictions['Prediction']
    predictions['Key in Train'] = predictions.apply(lambda x: x['Key'] in train_cm_map[x['Cipher']], axis=1)

    return predictions








