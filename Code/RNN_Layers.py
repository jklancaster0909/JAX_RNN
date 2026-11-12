import jax
import string
import jax.numpy as jnp
import jax.lax as lax
import jax.tree_util as jtu
from jaxtyping import PyTree, PRNGKeyArray

import numpy as np
import yaml

import ModelParams

class Layer:

    def __init__(self, key, key_W, shape: tuple[int, int]):

        self.key = key
        self.key_W = key_W
        self.shape = shape
        self.W = jax.random.uniform(key=self.key_W, shape=self.shape)

    def get_pytree(self):
        tree = {
            'key': self.key,
            'key_W': self.key_W,
            'shape': self.shape,
            'W': self.W
        }
        return tree
    
    def get_params(self):
        return {'W': self.W}
    
    def update_params(self, tree: PyTree):
        self.W = tree['W']
        

class StateLayer(Layer):

    def __init__(self, config: ModelParams, shape: tuple[int, int]):
        config.key, key_W, key_b = jax.random.split(config.key, 3)
        super().__init__(config.key, key_W, shape)

        self.key_b = key_b
        self.bias = jax.random.uniform(key=self.key_b, shape=(shape[0]))
        self.states = jnp.zeros(shape=(config.sequence_size, shape[0]))

    def get_pytree(self):
        tree = super().get_pytree()
        tree['key_b'] = self.key_b
        tree['bias'] = self.bias
        tree['states'] = self.states

    def get_params(self):
        tree = super().get_params()
        tree['bias'] = self.bias
        tree['states'] = self.states
        
        return tree
    
    def update_params(self, tree: PyTree):
        super().update_params(tree)
        self.bias = tree['bias']
        self.states = tree['states']


class HiddenLayer(StateLayer):

    def __init__(self, config: ModelParams):
        super().__init__(config, (config.hidden_size, config.hidden_size))


class OutputLayer(StateLayer):

    def __init__(self, config: ModelParams):
        super().__init__(config, (config.vocab_size, config.hidden_size))


class InputLayer(Layer):

    def __init__(self, config):
        config.key, key_W = jax.random.split(config.key, 2)
        super().__init__(config.key, key_W, (config.hidden_size, config.input_size))
        self.inputs = config.inputs

    def get_pytree(self):
        tree = super().get_pytree()
        tree['inputs'] = self.inputs
        return tree
    
    def get_params(self):
        tree = super().get_params()
        tree['inputs'] = self.inputs
        return tree
    
    def update_params(self, tree):
        super().update_params(tree)
        self.inputs = tree['inputs']



