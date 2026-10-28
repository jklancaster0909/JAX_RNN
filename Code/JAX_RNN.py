import jax
import string
import jax.numpy as jnp
import jax.lax as lax

import numpy as np


def string_to_one_hot(inputs: np.ndarray) -> jnp.ndarray:
    """
    Converts an array of char to a one-hot encoded array.
    Pulled from: https://medium.com/@thisislong/building-a-recurrent-neural-network-from-scratch-ba9b27a42856

    Parameters:
    -----------
    inputs: np.ndarray of chars to be converted

    Returns:
    --------
    jnp.ndarray
        A JAX ndarray of the one-hot encoded string
    """


    char_to_index = {char: i for i, char in enumerate(string.ascii_uppercase)}

    one_hot_inputs = []
    for row in inputs:
        one_hot_list = []
        for char in row:
            if char.upper() in char_to_index:
                one_hot_vector = np.zeros(len(string.ascii_uppercase))
                one_hot_vector[char_to_index[char.upper()]] = 1
                one_hot_list.append(one_hot_vector)
        one_hot_inputs.append(one_hot_list)

    return jnp.array(one_hot_inputs)

def load_data():
    """
    Defines inputs and expected datasets

    TODO: Add load for datasets
    """

    inputs = np.array([
        ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
        ["Z","Y","X","W","V","U","T","S","R","Q","P","O","N","M","L","K","J","I","H","G","F","E","D","C","B","A"],
        ["B","D","F","H","J","L","N","P","R","T","V","X","Z","A","C","E","G","I","K","M","O","Q","S","U","W","Y"],
        ["M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L"],
        ["H","G","F","E","D","C","B","A","L","K","J","I","P","O","N","M","U","T","S","R","Q","X","W","V","Z","Y"]
    ])

    expected = np.array([
        ["B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A"],
        ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
        ["C","E","G","I","K","M","O","Q","S","U","W","Y","A","B","D","F","H","J","L","N","P","R","T","V","X","Z"], 
        ["N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H","I","J","K","L","M"],
        ["I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","A","B","C","D","E","F","G","H"]
    ])

    return inputs, expected

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

def construct_model_pytree(config):
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
    input_layer, config = initialize_input_layer(config, one_hot_inputs[0])

    model = {
        'input_layer': input_layer,
        'hidden_layer': hidden_layer,
        'output_layer': output_layer,
        'config': config
    }

    return model

@jax.jit
def feed_forward(model):
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
def loss_fn(predictions, expected):
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
def compute_rnn_loss(model, expected):
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

if __name__ == "__main__":

    #Load data
    inputs, expected = load_data()

    #One hot encode data
    one_hot_inputs = string_to_one_hot(inputs)
    one_hot_expected = string_to_one_hot(expected)

    #Construct initial model pytree
    config = initialize_config(sequence_size=len(one_hot_inputs[0]))
    model = construct_model_pytree(config)

    #Define value and gradient function
    loss_grad_fn = jax.jit(jax.value_and_grad(compute_rnn_loss, has_aux=True, allow_int=True))

    #Run forward pass and compute gradients
    loss_and_model, gradients = loss_grad_fn(model, one_hot_expected[0])
    loss, model = loss_and_model

    print(gradients)
        





    