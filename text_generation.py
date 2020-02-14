import tensorflow as tf
import numpy as np
import operator


BATCH_SIZE = 16
SEQUENCE_LENGTH = 256


def split_input_target(chunk):
    input_text = chunk[:, :, :-1]
    target_text = chunk[:, :, 1:]
    return input_text, target_text


def one_hot_to_number(arr):
    return np.argmax(arr, axis=3)


def _one_hot(arr, total_characters):
    encoded = np.zeros((len(arr), total_characters), dtype=int)
    for idx, val in enumerate(arr):
        encoded[idx][val] = 1
    return encoded


def _text_to_input_array(text, character_to_number, total_characters):
    input_array = []
    for char in text:
        input_array.append(character_to_number[char])
    return _one_hot(input_array, total_characters)


def _generate_batches(input_text, batch_size, sequence_length, character_to_number, total_characters):
    blocklength = len(input_text)//batch_size
    batches=[]
    for i in range(0, blocklength, sequence_length):
        batch=[]
        for j in range(batch_size):
            start = j*blocklength+i
            end = min(start+sequence_length, j*blocklength + blocklength)
            one_hot_encoded = _text_to_input_array(input_text[start:end], character_to_number, total_characters)
            batch.append(one_hot_encoded)
        batches.append(np.array(batch, dtype=int))
    return batches


def get_input(book_name="The count of monte cristo.txt"):
    f = open(book_name)
    text = f.read().lower()

    character_map = {}
    for character in text:
        if character_map.get(character) is not None:
            current_count = character_map[character]
            character_map[character] = current_count + 1
        else:
            character_map[character] = 1

    sorted_map = sorted(character_map.items(), key=operator.itemgetter(1), reverse=True)
    THRESHOLD = 500  # Avoiding characters which occur less than '$THRESHOLD' number of times
    frequent_sorted_map = dict(filter(lambda v: v[1] >= THRESHOLD, sorted_map))
    rare_keys = dict(filter(lambda v: v[1] < THRESHOLD, sorted_map)).keys()
    for rare_char in rare_keys:
        text = text.replace(rare_char, '')

    character_to_number = {}
    for index, character in enumerate(frequent_sorted_map):
        character_to_number[character[0]] = index
    number_to_character = {v: k for k, v in character_to_number.items()}

    total_characters = len(frequent_sorted_map)
    print("Original Characters : {0}.       Frequent Characters : {1}".format(len(sorted_map), len(frequent_sorted_map)))
    batch_input = _generate_batches(text, BATCH_SIZE, SEQUENCE_LENGTH, character_to_number, total_characters)
    # Removing last incomplete sequence to maintain shape
    batch_input = np.array(batch_input[:-1], dtype=int)
    return batch_input, number_to_character, character_to_number


input_blocks, number_to_text_dict, character_to_number_dict = get_input()

total_characters = len(number_to_text_dict)

epochs = 7
batch_size = 16
learning_rate = 0.01
hidden_units = 256
sequence_length = 256

seed = 0
tf.reset_default_graph()
tf.set_random_seed(seed=seed)

X_train, Y_train = split_input_target(input_blocks)

X = tf.placeholder(shape=[None, None, total_characters], dtype=tf.float64)
Y = tf.placeholder(shape=[None, None, total_characters], dtype=tf.float64)
length = tf.placeholder(shape=[None], dtype=tf.int64)

cells = [tf.compat.v1.nn.rnn_cell.BasicLSTMCell(hidden_units), tf.nn.rnn_cell.GRUCell(hidden_units)]
stacked_rnn_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)
previous_state = stacked_rnn_cell.zero_state(batch_size, dtype=tf.float64)
rnn_outputs, final_state = tf.nn.dynamic_rnn(stacked_rnn_cell, X, sequence_length=length, initial_state=previous_state)

rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units])

Wout = tf.Variable(tf.truncated_normal(shape=(hidden_units, total_characters), stddev=0.1, dtype=tf.float64))
bout = tf.Variable(tf.zeros(shape=[total_characters], dtype=tf.float64))

Z = tf.matmul(rnn_outputs_flat, Wout) + bout
Y_flat = tf.reshape(Y, [-1, total_characters])

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_flat, logits=Z)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
session = tf.Session()
session.run(tf.global_variables_initializer())

lengths = [256] * 16
loss_history = []
for e in range(1, epochs + 1):
    print('Epoch: {0}'.format(e))
    for i in range(input_blocks.shape[0]):
        present_state = session.run([final_state],
                                    {X: X_train[i],
                                     length: lengths,
                                     })
        feed = {X: X_train[i],
                Y: Y_train[i],
                previous_state: present_state,
                length: lengths,
                }
        l, state_1 = session.run([loss, train], feed)
        if i % 1 == 0:
            loss_history.append(np.mean(l))
        if i % 50 == 0:
            print('Batch: {0}. Loss: {1}.'.format(i, np.mean(l)))

# Prediction
starting_strings = ['the', 'once']
sentences = []
for s in starting_strings:
    init_string = s
    one_hot_text = []

    for i in range(256 - len(init_string)):
        one_hot_text = np.zeros([16, 255, total_characters])

        for j, character in enumerate(init_string[::-1]):
            index = character_to_number_dict[character]
            j += 1
            one_hot_text[-1][-j][index] = 1

        present_state = session.run([final_state],
                                    {X: one_hot_text,
                                     length: lengths,
                                     })
        feed = {X: one_hot_text,
                previous_state: present_state,
                length: lengths}

        final_output = session.run([Z], feed)

        softmax = np.exp(final_output[-1][-1])
        softmax = softmax / np.sum(softmax)
        random_choice = np.random.choice(list(range(total_characters)), p=softmax)
        init_string += number_to_text_dict[random_choice]

    print("*"*16)
    print(init_string)
    print("*"*16)
    sentences.append(init_string)
