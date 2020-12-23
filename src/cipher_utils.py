import secretpy

from numpy import all
from numpy.random import shuffle, seed, randint

import base64
from Crypto.Cipher import AES


class SimpleEncoder():
    def __init__(self, encoding=None):
        if encoding is None or encoding == "":
            self.encode = self.decode = lambda x:x
        if encoding == 'b64':
            self.encode = base64.b64encode
            self.decode = base64.b64decode
        if encoding == 'b85':
            self.encode = base64.b85encode
            self.decode = base64.b85decode


class AESCipher():
    def __init__(self, encoding=None, iv=None):
        self.__iv = iv or b'5432109876543210'
        self.encoder = SimpleEncoder(encoding)

    def encrypt(self, text, key, alphabet=""):
        if isinstance(text, str):
            text = text.encode('utf-8')

        cipher = AES.new(key, AES.MODE_CBC, self.__iv)
        # cipher = AES.new(key, AES.MODE_CBC)
        ctx = cipher.encrypt(text)
        ctx = self.encoder.encode(ctx)
        # ctx = ctx.decode('utf-8')
        return ctx

    def decrypt(self, text, key, alphabet=""):
        ctx = self.encoder.decode(text)
        cipher = AES.new(key, AES.MODE_CBC, self.__iv)
        # cipher = AES.new(key, AES.MODE_CBC)
        ptx = cipher.decrypt(ctx)
        # ptx = ptx.decode('utf-8')
        return ptx


class FakeAES():
    """
    Fake AES Cipher that outputs a random sequence of strings from the alphabet of the same length as the input message.
    """
    def __init__(self, alphabet='', encoding=None):
        self.encoder = SimpleEncoder(encoding)
        self.alphabet = alphabet

    def encrypt(self, text):
        n = len(text)
        cipher = list(self.alphabet)
        shuffle(cipher)
        cipher = ''.join(cipher[:n])

        return cipher

    def decrypt(self):
        pass

    def set_alphabet(self, alphabet):
        self.alphabet = alphabet

    def set_key(self, key):
        seed(key)


def gen_cm(cipher, key, keep_ws=True):
    """ Generate a cipher machine for the cipher """
    cm = secretpy.CryptMachine(cipher, key)
    if not keep_ws:
        cm = secretpy.cmdecorators.NoSpaces(cm)
    return cm


def initialize_cipher_machines():
    """ Initialise a map of ciphers"""

    ciphers = {
        "plain": None,
        "caesar": gen_cm(secretpy.Caesar(), key=3),
        "columnar": gen_cm(secretpy.ColumnarTransposition(), key="monkey"),
        "vigenere": gen_cm(secretpy.Vigenere(), key="monkey"),
        "substitution": gen_cm(secretpy.SimpleSubstitution(), key=""),
        "fakeaes": FakeAES()
    }

    return ciphers


def generate_key_list(cipher, nkeys, alphabet):
    """ Generate a list of keys for the given cipher type. """
    if cipher in ["substitution"]:
        key = list(alphabet)
        keys = []
        for _ in range(nkeys):
            shuffle(key)
            keys.append(''.join(key))

    elif cipher in ["columnar", "vigenere"]:
        key = list(alphabet)
        keys = []
        for _ in range(nkeys):
            l = randint(1, len(alphabet))
            shuffle(key)
            keys.append(''.join(key[:l]))

    elif cipher in ['caesar', 'fakeaes']:
        keys = randint(1, len(alphabet), size=nkeys)
        keys = list(keys)

    else:
        raise ValueError('Unknown cipher map %s'%(cipher))

    return keys


def get_new_key(cipher, alphabet):
    """ Generate a new key for the given cipher type. """
    if cipher in ["substitution"]:
        key = list(alphabet)
        shuffle(key)
        key = ''.join(key)
    elif cipher in ["columnar", "vigenere"]:
        key = list(alphabet)
        shuffle(key)
        l = randint(1, len(alphabet))
        key = ''.join(key[:l])

    elif cipher in ['caesar', 'fakeaes']:
        key = randint(len(alphabet))

    else:
        raise ValueError('Unknown cipher map %s'%(cipher))

    return key


def test_decrypt(cipher_dict, cipher_map, cipher, messages_plain, num_test):
    """ Test whether ciphertexts can be correctly decrypted. """
    cm = cipher_map[cipher]
    messages_en, keys, _ = cipher_dict[cipher]

    # Check that first 100 messages get descrypted to correct raw message
    messages_de = []
    for m,k in zip(messages_en[:num_test], keys[:num_test]):
        cm.set_key(k)
        messages_de.append(cm.decrypt(m))

    print(all([m == md for m, md in zip(messages_plain[:num_test], messages_de[:num_test])]))