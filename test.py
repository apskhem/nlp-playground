from attacut import tokenize

from keras.models import load_model
from keras.layers import Input, Dense, LSTM, Bidirectional, StringLookup, Embedding
from keras.utils import to_categorical, set_random_seed
from keras_preprocessing import sequence
from gensim.models import FastText
from sklearn.metrics import confusion_matrix

import numpy as np

import utils

LM_FILE = "model/nlp_model.hdf5"
WV_FILE = "model/wv_model.hdf5"
INPUT_FILE = "rechal/input.txt"
OUTPUT_FILE = "rechal/ans.txt"
MAX_WORDS_COUNT = 92
WORD_VECTOR_SIZE = 128

print("reading files...")

input_text = utils.read_file(INPUT_FILE)

print("reading files... DONE")
print("parsing files...")

input_list = [line.split("::", 1)[1] for line in input_text.splitlines()]

ans_map_n2k = {0: 'H', 1: 'M', 2: 'P'}
ans_map_k2n = {'H': 0, 'M': 1, 'P': 2}

print("parsing files... DONE")
print("tokenizing words...")

sentences = [utils.filter_whitespace(tokenize(s)) for s in input_list]
max_word_count = utils.get_longest_word_count(sentences)

print("tokenizing words... DONE")
print("vectorizing words...")

wv_model = FastText.load(WV_FILE)

print("vectorizing words... DONE")
print("preparing model...")

model = load_model(LM_FILE)

model.summary()

x_pred = utils.prepare_training_set(sentences, wv_model, MAX_WORDS_COUNT, WORD_VECTOR_SIZE)

y_pred = model.predict(x_pred)
y_pred_idx = y_pred.argmax(axis=1)

output = ["::".join([str(idx + 1),  ans_map_n2k[x]]) for idx, x in enumerate(y_pred_idx)]
output = "\n".join(output)

utils.write_file(OUTPUT_FILE, output)
