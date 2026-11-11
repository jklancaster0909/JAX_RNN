import jax
import string
import jax.numpy as jnp
import jax.lax as lax
import jax.tree_util as jtu
from jaxtyping import PyTree, PRNGKeyArray

import numpy as np
import yaml

from RNN_Text_Utils import *


def initialize_config(sequence_size,
                      seed=1701, 
                      hidden_size=128,
                      vocab_size = len(string.ascii_uppercase),
                      alpha=0.001):
    """Initializes config pytree for the model

    Parameters:
    -----------
    sequence_size: int - length of the data chunck
    seed: int - seed for random number generator
    hidden_size: int - size of the hidden layer
    vocab_size: int - size of vocab - length of array for a single char
    alpha: float - learning rate

    Returns:
    --------
    PyTree - config metadata organized as a pytree
    """

    config = {
        'seed': seed,
        'key': jax.random.key(seed),
        'vocab_size': vocab_size,
        'sequence_size': sequence_size,
        'hidden_size': hidden_size,
        'alpha': alpha,
    }

    return config

#TODO: Create a general layer constructor 
def initialize_hidden_layer(config):

    config['key'], key_W, key_hb = jax.random.split(config['key'], 3)
    hidden_size=config['hidden_size']
    sequence_size=config['sequence_size']

    W = jax.random.uniform(key=key_W, shape=(hidden_size, hidden_size))
    h_bias = jax.random.uniform(key=key_hb, shape=(hidden_size))
    h_states = jnp.zeros(shape=(sequence_size, hidden_size))

    hidden_layer = {
        'W': W,
        'h_bias': h_bias,
        'h_states': h_states
    }

    return hidden_layer, config

def initialize_output_layer(config):
    config['key'], key_V, key_ob = jax.random.split(config['key'], 3)
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']
    sequence_size = config['sequence_size']

    V = jax.random.uniform(key=key_V, shape=(vocab_size, hidden_size))
    o_bias = jax.random.uniform(key=key_ob, shape=(vocab_size))
    o_states = jnp.zeros(shape=(sequence_size, vocab_size)) 

    output_layer = {
        'V': V,
        'o_bias': o_bias,
        'o_states': o_states,
    }

    return output_layer, config

def initialize_input_layer(config, inputs):
    config['key'], key_U = jax.random.split(config['key'], 2)
    vocab_size = config['vocab_size']
    hidden_size = config['hidden_size']

    U = jax.random.uniform(key=key_U, shape=(hidden_size, vocab_size))

    input_layer = {
        'inputs': inputs,
        'U': U
    }

    return input_layer, config

def construct_model_pytree(config: PyTree, inputs: jnp.ndarray) -> dict:
    """Builds layers of RNN from config and organizes the initialized model into a single pytree

    Parameters:
    -----------
    config: PyTree - config metadata used to build model

    Returns:
    --------
    model: PyTree - initialized model organized into a single pytree
    """

    hidden_layer, config = initialize_hidden_layer(config)
    output_layer, config = initialize_output_layer(config)
    input_layer, config = initialize_input_layer(config, inputs)

    model = {
        'input_layer': input_layer,
        'hidden_layer': hidden_layer,
        'output_layer': output_layer,
        'config': config
    }

    return model

@jax.jit
def feed_forward(model: PyTree) -> tuple[jnp.ndarray, jnp.ndarray]:
      """Runs a feed forward pass of the model

      Parameters:
      -----------
      model: PyTree - Contains layers and metadata for the model

      Returns:
      --------
      all_states: PyTree - hidden states and output states for the entire feed forward pass
      """

      #Feed forward implementation for a single input
      def feed_forward_step(carry, input):

            #We only need to pull the previous hidden state
            h_state, _ = carry
            U = model['input_layer']['U']
            W = model['hidden_layer']['W']
            h_bias = model['hidden_layer']['h_bias']
            V = model['output_layer']['V']
            o_bias = model['output_layer']['o_bias']

            #Calculated new hidden state
            weighted_input = jnp.dot(U, input)
            weighted_h_state = jnp.dot(W, h_state)
            weighted_sum = weighted_input + weighted_h_state + h_bias
            activation = jnp.tanh(weighted_sum) #new h_state

            #Calculate output
            output = jnp.dot(V, activation) + o_bias
            prediction = jax.nn.softmax(output)

            #Return new carry
            new_carry = (activation, prediction)
            return new_carry, new_carry

      #Initial hidden state and output state for carry
      carry_init = (model['hidden_layer']['h_states'][0], model['output_layer']['o_states'][0])

      #Use lax.scan() to loop through all inputs
      _, all_states = lax.scan(feed_forward_step, carry_init, model['input_layer']['inputs'])

      return all_states

@jax.jit
def loss_fn(predictions: jnp.ndarray, expected: jnp.ndarray) -> float:
    """Cross-Entropy Loss Function: computes the cross entropy loss for each time step and sums over all time steps.

    Parameters:
    -----------
    predictions: jnp.ndarray - model output states
    expected: jnp.ndarray - one-hot encoded expected output

    Returns:
    --------
    loss: float - computed summed cross-entropy loss
    """

    #Compute cross entropy loss for a single time step
    @jax.jit
    def loss_t_step(prediction, expected):
        return -jnp.sum(expected * jnp.log(prediction))
    
    #Sum over cross-entropy loss at all time steps
    return jnp.sum(jax.vmap(loss_t_step, in_axes=(0, 0))(predictions, expected))

@jax.jit
def compute_rnn_loss(model: PyTree, expected: jnp.ndarray) -> tuple[float, PyTree]:
    """Runs a feed forward pass of the model and computes the loss

    Parameters:
    -----------
    model: PyTree - Model parameters and metadata
    expected: jnp.ndarray - One-hot encoded expected outputs

    Returns:
    --------
    loss: float - computed cross-entropy loss from model output states and expected
    model: PyTree - Model Parameters and metadata after feed forward pass
    """
    
    #Run feedforward pass
    all_states = feed_forward(model)

    #Update model PyTree with final states
    model['hidden_layer']['h_states'], model['output_layer']['o_states'] = all_states

    #Compute loss
    loss = loss_fn(model['output_layer']['o_states'], expected)

    return loss, model



def learning_loop(model, inputs, expected):
    """
    TODO: Proper implementation of this
    """

    learning_rate = model['config']['alpha']

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
            return param_leaf - learning_rate * grad_leaf
        else:
            return param_leaf
        
    num_iter = 50 #len(inputs)
        
    for i in range(num_iter):

        #update inputs
        model['input_layer']['inputs'] = inputs[i]

        #Define value and gradient function
        loss_grad_fn = jax.jit(jax.value_and_grad(compute_rnn_loss, has_aux=True, allow_int=True))

        #Run forward pass and compute gradients
        loss_and_model, gradients = loss_grad_fn(model, expected[i])
        loss, model = loss_and_model

        model = jtu.tree_map(apply_gradient_step, model, gradients)

        if i % 10 == 0:
            print(f'Loss at iter {i}: {loss}')

    return model, i

def decode_one_hot(data: jnp.ndarray, chars_rev: dict) -> string:
    """
    Decode one hot encoded data into a string

    Parameters:
    -----------
    data: jnp.ndarray - 2d array of data of shape (seq_size, vocab_size)
    chars_rev: dict - all chars in dataset with idx as key and values 'char' and 'count'

    Returns:
    --------
    string - data decoded into string
    """

    decoded_data = ''

    for row in data:
        char = chars_rev[int(jnp.argmax(row))]['char']
        decoded_data += char

    return decoded_data

# 2. Convert JAX Arrays to standard Python types (lists/scalars)
def to_python_types(leaf):
    """
    Convert a leaf of a PyTree to a standard Python type for printing

    Parameters:
    -----------
    leaf: leaf of a pytree, any data type

    Returns:
    --------
    list - jax array converted to list, or original leaf
    """

    if isinstance(leaf, (jax.dtypes.prng_key, PRNGKeyArray)):
        return jax.random.key_data(leaf).tolist()
    elif isinstance(leaf, (jax.Array, jnp.ndarray)):
        # Convert to a standard Python scalar or list
        return leaf.tolist()
    return leaf




if __name__ == "__main__":

    #Load data
    path = 'Datasets/shakespeare.txt'

    seq_size = 30
    inputs, expected, chars_rev = load_data(path, seq_size)


    #Construct initial model pytree
    config = initialize_config(sequence_size=jnp.size(inputs, 1), vocab_size=jnp.size(inputs, 2))
    print(config['key'].dtype)
    model = construct_model_pytree(config, inputs=inputs[0])

    trained_model, final_idx = learning_loop(model, inputs, expected)

    trained_model['config']['key'] = jax.random.key_data(trained_model['config']['key']).tolist()

    trained_model_dict = jtu.tree_map(to_python_types, trained_model)

    with open('data.yml', "w", encoding="utf-8") as outfile:
        yaml.dump(trained_model_dict, outfile, default_flow_style=False)

    decoded_output = decode_one_hot(trained_model['output_layer']['o_states'], chars_rev)
    decoded_expected = decode_one_hot(expected[final_idx], chars_rev)

    print(f'decoded output: {decoded_output}')
    print(f'expected output: {decoded_expected}')

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

        





    