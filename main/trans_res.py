import os
import pickle
import numpy as np
from numpy import array
# import tflite_runtime.interpreter as tflite
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from googletrans import Translator

def load_pickle(name):
    with open(name, "rb") as handle:
        return pickle.load(handle)

def dump_pickle(name, file):
    with open(name, "wb") as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

wordtoix = load_pickle("media/model/dict/train_data_hin_word2ix.pickle")
ixtoword = load_pickle("media/model/dict/train_data_hin_ix2word.pickle")
train_data_eng_word2ix = load_pickle("media/model/dict/train_data_eng_word2ix.pickle")
train_data_eng_ix2word = load_pickle("media/model/dict/train_data_eng_ix2word.pickle")

m1 = load_model("media/model/hell_33.h5")

def greedySearch(text):
    max_length = 20
    # wordtoix = load_pickle("media/model/wordtoix.pickle")
    # ixtoword = load_pickle("media/model/ixtoword.pickle")
    wordtoix = load_pickle("media/model/dict/train_data_hin_word2ix.pickle")
    ixtoword = load_pickle("media/model/dict/train_data_hin_ix2word.pickle")

    in_text = '<'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
#         print(sequence)
        yhat = m1.predict([text,sequence], verbose=0, use_multiprocessing=True)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == '>':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ''.join(final)
    return final

def transliteration(text):
    output = ''
    for ele in text.split():
        if ele.isalnum() and not ele.isalpha():
            output += ' '+ele
            continue
        if ele.isupper():
            output += ' '+ele
            continue
        ele = ele.lower()
        if ele == '&':
            ele = 'and'
        temp = ele
        ele = [train_data_eng_word2ix[word] for word in list(ele) if word in train_data_eng_word2ix]
        ele = pad_sequences([ele], maxlen=21)
        output += ' ' + greedySearch(ele)
        if temp[-1] in '.!,':
            if temp[-1] == '.':
                output += '|'
            else : output += temp[-1]
    return output.strip()

def main(text, format):
    if format == 'translation':
        translator = Translator()
        out = translator.translate(text, dest='en').text
        return (out, )
    elif format == 'transliteration':
        return (transliteration(text), )
    else :
        translator = Translator()
        out = translator.translate(text, dest='en')
        return [transliteration(text), out.text]

if __name__ == '__main__':
    pass