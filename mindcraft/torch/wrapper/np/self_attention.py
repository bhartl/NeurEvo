from numpy import asarray
from numpy import array_equal
from numpy import sqrt as np_sqrt
from numpy import ndarray
from numpy import sum as np_sum
from numpy import exp as np_exp
from numpy import random as np_random
from typing import Union


class SelfAttentionMap(object):
    """ see https://attentionagent.github.io/

    (c) B. Hartl 2021
    """

    def __init__(self,
                 input_size: int, latent_size: Union[int, tuple],
                 key_matrix=None, key_bias=None,
                 query_matrix=None, query_bias=None,
                 col_softmax=True,
                 ):
        """ Constructs a SelfAttention Layer

        :param input_size:
        :param latent_size:
        """

        self.input_size = input_size
        self.latent_size = latent_size
        self.col_softmax = col_softmax

        self.key_matrix = key_matrix
        if self.key_matrix is None:
            self.key_matrix = np_random.randn(self.input_size, self.latent_size)
        self.key_matrix = asarray(self.key_matrix)

        self.key_bias = key_bias
        if self.key_bias is None:
            self.key_bias = np_random.randn(self.latent_size)
        self.key_bias = asarray(self.key_bias)
        
        assert array_equal(self.key_matrix.shape, (self.input_size, self.latent_size)), self.key_matrix.shape

        self.query_matrix = query_matrix
        if self.query_matrix is None:
            self.query_matrix = np_random.randn(self.input_size, self.latent_size)
        self.query_matrix = asarray(self.query_matrix)

        self.query_bias = query_bias
        if self.query_bias is None:
            self.query_bias = np_random.randn(self.latent_size)
        self.query_bias = asarray(self.query_bias)

        assert array_equal(self.query_matrix.shape, (self.input_size, self.latent_size)), self.query_matrix.shape

        self.one_over_sqrt_input_size = 1./np_sqrt(self.input_size)

    def key(self, x: Union[ndarray, object]) -> Union[ndarray, object]:
        x = asarray(x)
        return x.dot(self.key_matrix) + self.key_bias

    def query(self, x: Union[ndarray, object]) -> Union[ndarray, object]:
        x = asarray(x)
        return x.dot(self.query_matrix) + self.query_bias

    def attention_map(self, x: Union[ndarray, object]) -> Union[ndarray, object]:
        x = asarray(x)
        x = self.key(x).dot(self.query(x).T) / np_sqrt(self.input_size)  # correlation
        x = np_exp(x)                                                    # prepare softmax
        if self.col_softmax:
            return x / np_sum(x, axis=-1, keepdims=True)                 # col-wise normalized
        return x / np_sum(x, -2, keepdims=True)                          # row-wise normalized

    def forward(self, x: Union[ndarray, object]) -> object:
        if self.col_softmax:
            return self.attention_map(x)
        return self.attention_map(x).transpose((-1, -2))


class SelfAttention(SelfAttentionMap):
    """ A simple `numpy`-based self-attention layer

    (c) B. Hartl 2021
    """
    QUERY_KEY_VALUE_MODE = "qkv"
    QUERY_KEY_MODE = "qk"

    def __init__(self,
                 input_size: int, latent_size: Union[int, tuple],
                 key_matrix=None, key_bias=None,
                 query_matrix=None, query_bias=None,
                 value_matrix=None, value_bias=None,
                 col_softmax=True,
                 mode=QUERY_KEY_MODE
                 ):

        SelfAttentionMap.__init__(self,
                                  input_size=input_size, latent_size=latent_size,
                                  key_matrix=key_matrix, key_bias=key_bias,
                                  query_matrix=query_matrix, query_bias=query_bias,
                                  col_softmax=col_softmax)

        self._mode = None
        self.mode = (mode, value_matrix, value_bias)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        value_matrix, value_bias = None, None
        if isinstance(value, tuple):
            value, value_matrix, value_bias = value

        assert value in (self.QUERY_KEY_MODE, self.QUERY_KEY_VALUE_MODE)
        self._mode = value

        if value == self.QUERY_KEY_VALUE_MODE:
            self.value_matrix = value_matrix
            if self.value_matrix is None:
                self.value_matrix = np_random.randn(self.input_size, self.latent_size)
            self.value_matrix = asarray(self.value_matrix)

            self.value_bias = value_bias
            if self.value_bias is None:
                self.value_bias = np_random.randn(self.latent_size)
            self.value_bias = asarray(self.value_bias)

            assert array_equal(self.value_matrix.shape, (self.input_size, self.latent_size)), self.value_matrix.shape

    def value(self, x: Union[ndarray, object]) -> Union[ndarray, object]:
        assert self.mode
        x = asarray(x)
        return x.dot(self.value_matrix) + self.value_bias

    def forward(self, x: Union[ndarray, object]) -> object:
        attention_map = self.attention_map(x)
        if self.mode == self.QUERY_KEY_VALUE_MODE:
            x = self.value(x)

        if self.col_softmax:
            return attention_map.dot(x)

        return attention_map.transpose((-1, -2)).dot(x)
