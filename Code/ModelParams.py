import jax
import string
import jax.numpy as jnp
import jax.lax as lax
import jax.tree_util as jtu
from jaxtyping import PyTree, PRNGKeyArray


import numpy as np
import yaml
import time

from RNN_Layers import HiddenLayer, OutputLayer, InputLayer
from JAX_RNN import *
from RNN_Text_Utils import *

class ModelParams:

    def __init__(self,
                 inputs,
                 expected,
                 seed=1701, 
                 hidden_size=128,
                 alpha=0.01):
        
        self.inputs = inputs
        self.expected = expected
        self.sequence_size=jnp.size(self.inputs, 1)
        self.vocab_size=jnp.size(self.inputs, 2)
        self.seed = seed
        self.key_root = jax.random.key(self.seed)
        self.key = self.key_root
        self.hidden_size = hidden_size
        self.alpha = alpha

        self.partition_dataset()

        self.hidden_layer = HiddenLayer(self.key, 
                                        self.hidden_size, 
                                        self.sequence_size)
        
        #Initialize with training expected
        self.output_layer = OutputLayer(self.hidden_layer.key, 
                                        self.vocab_size, 
                                        self.hidden_size, 
                                        self.sequence_size, 
                                        self.training_expected)
        
        #Initialize with training inputs
        self.input_layer = InputLayer(self.output_layer.key, 
                                      self.hidden_size, 
                                      self.vocab_size, 
                                      self.training_inputs)
        
        self.key = self.input_layer.key

    def partition_dataset(self):

        print('Partitioning dataset...')

        #Shuffle dataset
        self.key, subkey = jax.random.split(self.key)
        length = jnp.size(self.inputs, 0)
        idxs = jax.random.permutation(subkey, length).astype(jnp.int32)
        shuffled_inputs = self.inputs[idxs]
        shuffled_expected = self.expected[idxs]

        #Get dataset sizes
        #TODO: make this more flexible
        train_size = jnp.ceil(length*0.7).astype(jnp.int32)
        valid_size = jnp.ceil((length-train_size)/2).astype(jnp.int32)

        #Create, training, validation, and test datasets
        self.training_inputs = shuffled_inputs[:train_size]
        self.training_expected = shuffled_expected[:train_size]
        self.valid_inputs = shuffled_inputs[train_size:train_size+valid_size]
        self.valid_expected = shuffled_expected[train_size:train_size+valid_size]
        self.test_inputs = shuffled_inputs[train_size+valid_size:]
        self.test_expected = shuffled_expected[train_size+valid_size:]

        print('Dataset partitioned\n')


        



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

    def learning_loop(self):
        """
        TODO: Proper implementation of this
        """

        def apply_gradient_step(param_leaf, grad_leaf):
            """
            Applies the gradient step for a leaf of the model PyTree

            Parameters:
            -----------
            paramleaf: leaf of the model PyTree
            gradleaf: leaf of thr gradient PyTree

            returns:
            --------
            new leaf with gradient applies

            """


            if isinstance(param_leaf, (jnp.ndarray, jax.Array)) and isinstance(grad_leaf, (jnp.ndarray, jax.Array)):
                return param_leaf - self.alpha * grad_leaf
            else:
                return param_leaf
        
        num_epochs = 1
        batch_size = len(self.input_layer.inputs)
        print(f'Batch size: {batch_size}\n')
        params = self.get_params()
        for epoch in range(num_epochs):
            
            self.key, subkey = jax.random.split(self.key)
            length = jnp.size(self.input_layer.inputs, 0)
            idxs = jax.random.permutation(subkey, length).astype(jnp.int32)
            self.input_layer.inputs = self.input_layer.inputs[idxs]
            self.output_layer.expected = self.output_layer.expected[idxs]
            
            for i in range(batch_size):

                #update inputs
                params['input_layer']['inputs'] = self.input_layer.inputs[i]
                params['output_layer']['expected'] = self.output_layer.expected[i]

                #Define value and gradient function
                loss_grad_fn = jax.jit(jax.value_and_grad(compute_rnn_loss, has_aux=True))

                #Run forward pass and compute gradients
                loss_and_params, gradients = loss_grad_fn(params)
                loss, params = loss_and_params

                params = jtu.tree_map(apply_gradient_step, params, gradients)

                if i % 1000 == 0:
                    print(f'Loss at iter {i}: {loss}')

        return params, i


if __name__ == "__main__":

    #Load data
    path = 'Datasets/shakespeare.txt'

    seq_size = 30
    inputs, expected, chars_rev = load_data(path, seq_size)


    #Construct initial model pytree
    #config = initialize_config(sequence_size=jnp.size(inputs, 1), vocab_size=jnp.size(inputs, 2)
    #model = construct_model_pytree(config, inputs=inputs[0])

    model = ModelParams(inputs=inputs, expected=expected)
    #model_tree = model.get_pytree()

    #print(model.hidden_layer.states)
    #print(model_tree['hidden_layer']['states'])


    #trained_model, final_idx = learning_loop(model_tree, inputs, expected)
    trained_model, final_idx = model.learning_loop()

    #trained_model['config']['key'] = jax.random.key_data(trained_model['config']['key']).tolist()

    #trained_model_dict = jtu.tree_map(to_python_types, trained_model)

    model.update_params(trained_model)
    trained_model_dict = model.get_pytree(python_type=True)

    with open('data.yml', "w", encoding="utf-8") as outfile:
        yaml.dump(trained_model_dict, outfile, default_flow_style=False)

    decoded_output = decode_one_hot(trained_model['output_layer']['states'], chars_rev)
    decoded_expected = decode_one_hot(expected[final_idx], chars_rev)



    print(f'decoded output: {decoded_output}')
    print(f'expected output: {decoded_expected}')
    print(len(decoded_output), len(decoded_expected))

    #print(trained_model_dict)

    """

    idx = 4

    #Define value and gradient function
    loss_grad_fn = jax.jit(jax.value_and_grad(compute_rnn_loss, has_aux=True, allow_int=True))

    #Run forward pass and compute gradients
    loss_and_model, gradients = loss_grad_fn(model, expected[idx])
    loss, model = loss_and_model
    """



    #print(gradients)

        





    