import jax
import string
import jax.numpy as jnp
import jax.lax as lax
import jax.tree_util as jtu
from jaxtyping import PyTree, PRNGKeyArray


import numpy as np
import yaml

from RNN_Layers import HiddenLayer, OutputLayer, InputLayer

class ModelParams:

    def __init__(self,
                 inputs,
                 sequence_size,
                 vocab_size = len(string.ascii_uppercase),
                 seed=1701, 
                 hidden_size=128,
                 alpha=0.01):
        
        self.inputs = inputs
        self.sequence_size = sequence_size
        self.seed = seed
        self.key = jax.random.key(self.seed)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.alpha = alpha

        self.hidden_layer = HiddenLayer(self)
        self.output_layer = OutputLayer(self)
        self.input_layer = InputLayer(self)


    def get_config(self):
        config = {
            'seed': self.seed,
            'key': self.key,
            'vocab_size': self.vocab_size,
            'sequence_size': self.sequence_size,
            'hidden_size': self.hidden_size,
            'alpha': self.alpha,
        }

        return config


