import jax
import string
import jax.numpy as jnp
import jax.lax as lax
import jax.tree_util as jtu
from jaxtyping import PyTree, PRNGKeyArray

import numpy as np
import yaml

def load_text(file_name: string, seq_size: int = 30):
    """
    Loads a text dataset as a list of strings

    Parameters:
    -----------
    file_name: string - path to the file

    Returns:
    --------
    list - list of strings representing lines of text
    """

    #Shakespeare: 85 unique chars
    raw_text = []
    with open(file_name, "r", encoding="utf-8") as file:

        #TODO: Better define number of sequences
        for i in range(100):
            raw_text.append(file.read(seq_size))

    print('text loaded\n')

    print(raw_text)
    
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

    Returns:
    --------
    char_idx: int - index of the next char added to chars
    """

    for char in line:
        if char in chars:
            chars[char]['count'] += 1
            chars_rev[chars[char]['idx']]['count'] += 1
            pass
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
    
    return char_idx
    


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

    #raw_text[0] = raw_text[0][1:]

    chars = {}
    chars_rev = {}
    char_idx = 0
    for i in range(len(raw_text)):
        raw_text[i] = list(raw_text[i])
        char_idx = update_chars_dict(raw_text[i], chars, chars_rev, char_idx)

    print('chars dictionary created')
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

    index_to_char = {i: char for i, char in enumerate(string.ascii_uppercase)}

    return one_hot_inputs, one_hot_expected, index_to_char


@jax.jit
def get_expected(input: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:

    seq_size=jnp.size(input, axis=0)

    new_input = lax.dynamic_slice_in_dim(input, 0, seq_size-1, axis=0)
    expected = lax.dynamic_slice_in_dim(input, 1, seq_size-1, axis=0)
    return new_input, expected

def load_data(file_name: string = None, seq_size: int = 30):
    """
    Defines inputs and expected datasets

    TODO: Add load for datasets
    """

    if file_name == None:
        return get_default_data()

    raw_text = load_text(file_name, seq_size)[:100]

    #max = get_max_line(raw_text)
    chars_dict, chars_rev = process_text(raw_text, max)

    one_hot_inputs = chars_to_one_hot(raw_text, chars_dict)

    vmap_get_expected = jax.jit(jax.vmap(get_expected, in_axes=0, out_axes=(0,0)))
    inputs, expected = vmap_get_expected(one_hot_inputs)

    print('Expected outputs created')
    print(f'Inputs shape: {inputs.shape}')
    print(f'Expected Shape{expected.shape}\n')

    return inputs, expected, chars_rev

