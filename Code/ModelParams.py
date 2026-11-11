import jax
import string
import jax.numpy as jnp
import jax.lax as lax
import jax.tree_util as jtu
from jaxtyping import PyTree, PRNGKeyArray

import numpy as np
import yaml

class ModelParams:

    def __init__(self,
                 sequence_size,
                 vocab_size = len(string.ascii_uppercase),
                 seed=1701, 
                 hidden_size=128,
                 alpha=0.01):
        
        self.sequence_size = sequence_size
        self.seed = seed
        self.key = jax.random.key(self.seed)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.alpha = alpha

        self.config = {
            'seed': self.seed,
            'key': self.key,
            'vocab_size': self.vocab_size,
            'sequence_size': self.sequence_size,
            'hidden_size': self.hidden_size,
            'alpha': self.alpha,
        }

        self.params = {
            'config': self.config
        }

    def set_key(self, key):
        self.key = key
        self.config['key'] = self.key


    """
    TODO:
    def set_hidden_layer(layer: Layer):
        hidden_layer = {
            'W': layer.W,
            'h_bias': h_bias,
            'h_states': h_states
        }
     """


class Layer:

    def __init__(self, key, key_W, shape: tuple[int, int]):

        self.key = key
        self.key_W = key_W
        self.shape = shape
        self.W = jax.random.uniform(key=self.key_W, shape=self.shape)
        

class HiddenLayer(Layer):

    def __init__(self, config: ModelParams):

        key, key_W, key_hb = jax.random.split(config.key, 3)
        super().__init__(key, key_W, (config.hidden_size, config.hidden_size))
        config.set_key(self.key)

        self.key_hb = key_hb
        self.h_bias = jax.random.uniform(key=self.key_hb, shape=(config.hidden_size))
        self.h_states = jnp.zeros(shape=(config.sequence_size, config.hidden_size))




