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


    def get_config(self, python_type: bool = False):
        config = {
            'seed': self.seed,
            'vocab_size': self.vocab_size,
            'sequence_size': self.sequence_size,
            'hidden_size': self.hidden_size,
            'alpha': self.alpha,
        }

        if python_type:
            config['key'] = jax.random.key_data(self.key).tolist()
        else:
            config['key'] = self.key

        return config
    
    def get_params(self):
        params = {
            'hidden_layer': self.hidden_layer.get_params(),
            'output_layer': self.output_layer.get_params(),
            'input_layer': self.input_layer.get_params()
        }
        return params

    def get_pytree(self, python_type: bool = False):
        tree = {
            'config': self.get_config(python_type=python_type),
            'hidden_layer': self.hidden_layer.get_pytree(python_type=python_type),
            'output_layer': self.output_layer.get_pytree(python_type=python_type),
            'input_layer': self.input_layer.get_pytree(python_type=python_type)
        }
        return tree
    
    def update_params(self, tree: PyTree):
        self.inputs = tree['input_layer']['inputs']
        self.input_layer.update_params(tree['input_layer'])
        self.hidden_layer.update_params(tree['hidden_layer'])
        self.output_layer.update_params(tree['output_layer'])


