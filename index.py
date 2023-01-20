from attacut import tokenize

from keras.models import Model
from keras.layers import Input, Dense, LSTM, GRU, Bidirectional, StringLookup, Embedding
from keras.utils import to_categorical, set_random_seed
from keras_preprocessing import sequence
from gensim.models import FastText
from sklearn.metrics import confusion_matrix

import numpy as np

import utils

IS_MODEL_SAVED_AFTER_TRAINING = True
IS_WV_SAVED_AFTER_TRAINING = True
SEED = 55 # unused
WORD_VECTOR_SIZE = 128

print("reading files...")

input_text = utils.read_file("data/input.txt")
c_input_text = utils.read_file("data/c_input.txt")
ans_text = utils.read_file("data/ans.txt")
stop_words_text = utils.read_file("data/stop_words.txt")

print("reading files... DONE")
print("parsing files...")

stop_words = stop_words_text.splitlines()
input_list = [line.split("::", 1)[1] for line in input_text.splitlines()]
ans_key_list = [line.split("::", 1)[1] for line in ans_text.splitlines()]

uniq_ans = list(set(ans_key_list))
uniq_ans.sort()

ans_map_n2k = {v: k for v, k in enumerate(uniq_ans)}
ans_map_k2n = {k: v for v, k in enumerate(uniq_ans)}

ans_num_list = [ans_map_k2n[x] for x in ans_key_list]

print("parsing files... DONE")
print("tokenizing words...")

sentences = [line.split("|") for line in c_input_text.splitlines()]
# sentences = [utils.filter_whitespace(tokenize(s)) for s in input_list]
sentences = [utils.filter_stop_words(s, stop_words) for s in sentences]
max_word_count = utils.get_longest_word_count(sentences)

# c = ["|".join(a) for a in sentences]
# utils.write_file("data/c_input.txt", "\n".join(c))

print("tokenizing words... DONE")
print("vectorizing words...")

# wv_model = FastText.load("model/wv_model.hdf5")

wv_model = FastText(
  sentences=sentences,
  vector_size=WORD_VECTOR_SIZE,
  window=3,
  min_count=1,
  epochs=16,
  sg=1,
)

if IS_WV_SAVED_AFTER_TRAINING:
  wv_model.save("model/wv_model.hdf5")

print("vectorizing words... DONE")
print("preparing model...")

input_size = max_word_count
output_size = len(uniq_ans)
rnn_size = (input_size + output_size) * 2 // 3

model_input = Input(shape=(input_size,WORD_VECTOR_SIZE))
model_rnn_1 = LSTM(rnn_size, activation="tanh", return_sequences=False)(model_input)
model_output = Dense(output_size, activation="softmax")(model_rnn_1)
model = Model(inputs=model_input, outputs=model_output)

model.compile(
  optimizer="adam",
  loss="categorical_crossentropy",
  metrics=["accuracy"]
)

# model.load_weights("model/nlp_model.hdf5")

model.summary()

print("preparing model... DONE")
print("training model...")

x_train = utils.prepare_training_set(sentences, wv_model, max_word_count, WORD_VECTOR_SIZE)
y_train = to_categorical(ans_num_list)

model.fit(
  x_train,
  y_train,
  epochs=25,
  batch_size=2,
  validation_split=0.1
)

if IS_MODEL_SAVED_AFTER_TRAINING:
  model.save("model/nlp_model.hdf5")

print("training model...DONE")

# y_pred = model.predict(x_train[2000:])

# print(y_pred)
# print(y_pred.argmax(axis=1))

# cm = confusion_matrix(labels[240], y_pred.argmax(axis=1))

# print("Confustion Matrix")
# print(cm)