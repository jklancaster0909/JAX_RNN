import jax
import jax.numpy as jnp
from jaxtyping import PyTree, PRNGKeyArray

class Layer:

    def __init__(self, key, key_W, shape: tuple[int, int]):

        self.key = key
        self.key_W = key_W
        self.shape = shape
        self.W = jax.random.uniform(key=self.key_W, shape=self.shape)

    def get_pytree(self, python_type: bool = False):
        
        if python_type:
            tree = {
                'key': jax.random.key_data(self.key).tolist(),
                'key_W': jax.random.key_data(self.key_W).tolist(),
                'W': self.W.tolist()
            }
        else:
            tree = {
                'key': self.key,
                'key_W': self.key_W,
                'W': self.W
            }

        tree['shape'] = self.shape

        
        return tree
    
    def get_params(self):
        return {'W': self.W}
    

    
    def update_params(self, tree: PyTree):
        self.W = tree['W']
        

class StateLayer(Layer):

    def __init__(self, key, shape: tuple[int, int], sequence_size: int):
        key_new, key_W, key_b = jax.random.split(key, 3)
        super().__init__(key_new, key_W, shape)

        self.key_b = key_b
        self.bias = jax.random.uniform(key=self.key_b, shape=(shape[0]))
        self.states = jnp.zeros(shape=(sequence_size, shape[0]))

    def get_pytree(self, python_type: bool = False):
        tree = super().get_pytree(python_type=python_type)

        if python_type:
            tree['key_b'] = jax.random.key_data(self.key_b).tolist()
            tree['bias'] = self.bias.tolist()
            tree['states'] = self.states.tolist()
        else:
            tree['key_b'] = self.key_b
            tree['bias'] = self.bias
            tree['states'] = self.states

        return tree

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

    def __init__(self, key, hidden_size: int, sequence_size:int ):
        super().__init__(key, (hidden_size, hidden_size), sequence_size)


class OutputLayer(StateLayer):

    def __init__(self, key, vocab_size: int, hidden_size: int, sequence_size: int, expected: jnp.ndarray):
        super().__init__(key, (vocab_size, hidden_size), sequence_size)
        self.expected = expected[0]

    def get_pytree(self, python_type: bool = False):
        tree = super().get_pytree(python_type=python_type)

        if python_type:
            tree['expected'] = self.expected.tolist()
        else:
            tree['expected'] = self.expected
        return tree
    
    def get_params(self):
        tree = super().get_params()
        tree['expected'] = self.expected
        return tree
    
    def update_params(self, tree):
        super().update_params(tree)
        self.expected = tree['expected']


class InputLayer(Layer):

    def __init__(self, key, hidden_size: int, vocab_size: int, inputs: jnp.ndarray):
        key_new, key_W = jax.random.split(key, 2)
        super().__init__(key_new, key_W, (hidden_size, vocab_size))
        self.inputs = inputs[0]

    def get_pytree(self, python_type: bool = False):
        tree = super().get_pytree(python_type=python_type)

        if python_type:
            tree['inputs'] = self.inputs.tolist()
        else:
            tree['inputs'] = self.inputs
        return tree
    
    def get_params(self):
        tree = super().get_params()
        tree['inputs'] = self.inputs
        return tree
    
    def update_params(self, tree):
        super().update_params(tree)
        self.inputs = tree['inputs']

    



