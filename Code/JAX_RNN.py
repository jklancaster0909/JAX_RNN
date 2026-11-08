import jax
import string
import jax.numpy as jnp
import jax.lax as lax
import jax.tree_util as jtu
from jaxtyping import PyTree, PRNGKeyArray

import numpy as np
import yaml


def load_text(file_name=None):
    """
    Loads a text dataset as a list of strings



    """

    #Shakespeare: 85 unique chars
    with open(file_name, "r", encoding="utf-8") as file:
        raw_text = file.readlines()

    print('text loaded\n')
    
    return raw_text

def get_max_line(raw_text: list) -> int:
    """
    Returns the length of the longest line in the raw text dataset

    Parameters:
    -----------
    raw_text: list - list of strings

    Returns:
    --------
    int - length of the longest string

    """

    raw_text[0] = raw_text[0][1:]

    max=0
    for i in range(len(raw_text)):

        #Remove \n from the end of each line
        if raw_text[i][-1] == '\n':
            raw_text[i] = raw_text[i][:-1]

        #Update max length
        length = len(raw_text[i])
        if length > max:
            max = length

    return max

def update_chars_dict(line: list, chars: dict, chars_rev: dict, char_idx: int):
    """
    Update chars dictionary for a line of text

    Parameters:
    -----------
    line: list - list of chars representing a line of text
    chars: dict - all chars in dataset with assigned index 'idx' and count value 'count'
    chars_rev: dict - all chars in dataset with idx as key and values 'char' and 'count'
    char_idx: int - index of the next char added to chars
    """

    for char in line:
        if char in chars:
            chars[char]['count'] += 1
            chars_rev[chars[char]['idx']]['count'] += 1
        else:
            chars[char] = {
                'idx': char_idx,
                'count': 1
            }
            chars_rev[char_idx] = {
                'char': char,
                'count': 1
            }
            char_idx += 1
    


def process_text(raw_text: list, max: int) -> tuple[dict, dict]:
    """
    Adds spaces to the end of each line to equal max length.
    Converts each line to a list of chars.
    Creates a dictionary with an assigned index and count value for each char.

    Parameters:
    -----------
    raw_text: list - list of strings
    max: int - length of the longest string

    Returns:
    --------
    dict - all chars in dataset with assigned index 'idx' and count value 'count'
    dict -  all chars in dataset with idx as key and values 'char' and 'count'
    """

    chars = {}
    chars_rev = {}
    char_idx = 0
    for i in range(len(raw_text)):
        raw_text[i] = list(raw_text[i] + ' ' * (max - len(raw_text[i])))
        update_chars_dict(raw_text[i], chars, chars_rev, char_idx)

    print('chars dictionary created')
    print(f'max length: {max}')
    print(f'vocab size: {len(chars)}\n')
        
    return chars, chars_rev

def chars_to_one_hot(inputs: np.ndarray, chars_dict: dict) -> jnp.ndarray:
    """
    Converts an array of char to a one-hot encoded array.

    Parameters:
    -----------
    inputs: np.ndarray - Chars to be converted
    chars_dict: dict - dictionary of chars in dataset with indices 'idx'

    Returns:
    --------
    jnp.ndarray - A JAX ndarray of the one-hot encoded chars datset
    """

    length = len(inputs)
    seq_size = len(inputs[0])
    vocab_size = len(chars_dict)



    one_hot_inputs = np.zeros((length, seq_size, vocab_size))
    for i, row in enumerate(inputs):
        if i % 10000 == 0:
            print(f'{i} rows encoded...')

        for j, char in enumerate(row):
            one_hot_inputs[i, j, chars_dict[char]['idx']] = 1

    print('\nInputs one hot encoded')
    one_hot_inputs = jnp.array(one_hot_inputs)
    print(f'Dataset shape: {one_hot_inputs.shape}\n')

    return one_hot_inputs

def string_to_one_hot(inputs: np.ndarray) -> jnp.ndarray:
    """
    Converts an array of char to a one-hot encoded array.
    Pulled from: https://medium.com/@thisislong/building-a-recurrent-neural-network-from-scratch-ba9b27a42856

    Parameters:
    -----------
    inputs: np.ndarray - Chars to be converted

    Returns:
    --------
    jnp.ndarray - A JAX ndarray of the one-hot encoded string
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

def get_default_data():
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

    #One hot encode data
    one_hot_inputs = string_to_one_hot(inputs)
    one_hot_expected = string_to_one_hot(expected)

    return one_hot_inputs, one_hot_expected

@jax.jit
def get_expected(input: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:

    seq_size=jnp.size(input, axis=0)

    new_input = lax.dynamic_slice_in_dim(input, 0, seq_size-1, axis=0)
    expected = lax.dynamic_slice_in_dim(input, 1, seq_size-1, axis=0)
    return new_input, expected

def load_data(file_name=None):
    """
    Defines inputs and expected datasets

    TODO: Add load for datasets
    """

    if file_name == None:
        return get_default_data()

    raw_text = load_text(file_name)[:100]

    max = get_max_line(raw_text)
    chars_dict, chars_rev = process_text(raw_text, max)

    one_hot_inputs = chars_to_one_hot(raw_text, chars_dict)

    vmap_get_expected = jax.jit(jax.vmap(get_expected, in_axes=0, out_axes=(0,0)))
    inputs, expected = vmap_get_expected(one_hot_inputs)

    print('Expected outputs created')
    print(f'Inputs shape: {inputs.shape}')
    print(f'Expected Shape{expected.shape}\n')

    return inputs, expected, chars_rev





        

    

    

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

    learning_rate = model['config']['alpha']

    def apply_gradient_step(param_leaf, grad_leaf):
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

    decoded_data = ''

    for row in data:
        char = chars_rev[int(jnp.argmax(row))]['char']
        decoded_data += char

    return decoded_data

# 2. Convert JAX Arrays to standard Python types (lists/scalars)
def to_python_types(leaf):
    if isinstance(leaf, (jax.dtypes.prng_key, PRNGKeyArray)):
        return jax.random.key_data(leaf).tolist()
    elif isinstance(leaf, (jax.Array, jnp.ndarray)):
        # Convert to a standard Python scalar or list
        return leaf.tolist()
    return leaf




if __name__ == "__main__":

    #Load data
    path = 'Datasets/shakespeare.txt'
    inputs, expected, chars_rev = load_data(path)


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

    print(decoded_output)
    print(decoded_expected)

    print(trained_model_dict)

    """

    idx = 4

    #Define value and gradient function
    loss_grad_fn = jax.jit(jax.value_and_grad(compute_rnn_loss, has_aux=True, allow_int=True))

    #Run forward pass and compute gradients
    loss_and_model, gradients = loss_grad_fn(model, expected[idx])
    loss, model = loss_and_model
    """



    #print(gradients)

        





    