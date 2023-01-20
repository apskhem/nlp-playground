import numpy as np

def filter_whitespace(segments):
  return [token.strip(" ") for token in segments if token.strip(" ")]

def filter_stop_words(sentence_words, stop_words):
  out = []

  for w in sentence_words:
    if w not in stop_words:
      out.append(w)

  return out

def get_longest_word_count(sentences):
  list_len = [len(w) for w in sentences]
  
  return max(list_len)

def read_file(path):
  f = open(path, mode="r", encoding="utf-8")

  content = f.read()

  f.close()

  return content

def write_file(path, content):
  f = open(path, mode="w")

  f.write(content)

  f.close()

def prepare_training_set(sentences, wv_model, max_sentence_len, vector_size):
  res = np.zeros((len(sentences), max_sentence_len, vector_size))

  s_idx = 0
  for s in sentences:
    w_idx = 0
    for word in s:
      if wv_model.wv.key_to_index.get(word) != None:
        res[s_idx, max_sentence_len - 1 - w_idx, :] = wv_model.wv[word]
    
      w_idx = w_idx + 1

    s_idx = s_idx + 1

  return res
