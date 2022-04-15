import string
import traceback
import re
import os
import json
import pickle
import glob
from PIL import Image
from time import time
import shutil
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz, process
import itertools

# from sklearn.model_selection import train_test_split

# import cv2
import numpy as np
from numpy import array
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf
# import tensorflow.compat.v1
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from keras.layers.embeddings import Embedding
from keras.layers.merge import add
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Attention, Layer,
                                     Add,concatenate,
                                     Input, 
                                     Dense,  
                                     LSTM, Bidirectional, GRU,
                                     ZeroPadding2D, 
                                     Convolution2D, Conv2D, 
                                     GlobalAveragePooling2D, GlobalAvgPool2D, GlobalMaxPooling2D, GlobalMaxPool2D, 
                                     AveragePooling2D, AvgPool2D, MaxPooling2D, MaxPool2D,
                                     Flatten,
                                     BatchNormalization, 
                                     Dropout)

from tensorflow.keras.layers import (Activation, 
                                     ReLU, 
                                     LeakyReLU, 
                                     Softmax)

from tensorflow.keras.optimizers import (SGD,
                                         Adam,
                                         Adagrad,
                                         Adadelta,
                                         RMSprop,
                                         Nadam)

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

# from tensorflow.keras.applications.vgg16 import VGG16

# from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16
# import pyphen
# import eng_to_ipa as ipa  # international phonetic aphabet

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
# from cStringIO import StringIO
from io import StringIO

# pd.set_option('max_columns', None)
# pd.set_option('display.max_colwidth', None)
pd.options.display.max_columns = 999
pd.options.display.max_colwidth = 999

def load_pickle(name):
    with open(name, "rb") as handle:
        return pickle.load(handle)
    
def dump_pickle(name, file):
    with open(name, "wb") as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
#     device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)

    with open(path, 'rb') as fp:
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        caching = True
        pagenos = set()

        for page in PDFPage.get_pages(fp, pagenos, password=password,caching=caching, check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue()

    device.close()
    retstr.close()
    return text

root = "./media/uploads/pdf"
df = pd.DataFrame(columns=["name", "content", "token"])
token_id = {}
id_token = {}
dump_pickle('./media/uploads/data/token_id.pkl', token_id)
dump_pickle('./media/uploads/data/id_token.pkl', id_token)
def token_id_id():
    return token_id, id_token

def get_token(content):
    print(content)
    ele = content.split()
    t = []
    for s in ele:
        if re.findall("[0-9a-zA-Z,\|!@#$%&*'\"\)\(]", s):
            continue
        for sy in "।|“”\"'":
            s = s.replace(sy, "")
        s = s.replace("\x00", "")
        t.append(s)
    return t

def create_df():
    global df
    global token_id
    global id_token
    print("amits")
    pdfs = os.listdir(root)
    for i, pdf in enumerate(pdfs):
        content = convert_pdf_to_txt(os.path.join(root, pdf))
        token = get_token(content)
        df = pd.concat([df, pd.DataFrame({"name" : pdf, "content" : content, 'token' : [token]}, index=[i])], axis=0)
        id_token[i] = token
        for ele in token:
            if token_id.get(ele) is None:
                token_id[ele] = {}
                token_id[ele][i] = 0
            if token_id[ele].get(i) is None:
                token_id[ele][i] = 0
            token_id[ele][i] += 1 
    dump_pickle('./media/uploads/data/token_id.pkl', token_id)
    dump_pickle('./media/uploads/data/id_token.pkl', id_token)
    return df

def add_pdf(url):
    global df
    token_id = load_pickle('./media/uploads/data/token_id.pkl')
    id_token = load_pickle('./media/uploads/data/id_token.pkl')
    # df = create_df()
    source_folder = "./media/uploads/image"
    file_name = os.listdir(source_folder)[0]
    print("see me", df)
    if df[df.name == file_name].shape[0] != 0:
        return "already exist"
    print(file_name)
    # construct full file path
    source = source_folder + "/"+ file_name
    destination = root +"/" + file_name
    # copy only files
    if os.path.isfile(source):
        shutil.copy(source, destination)
        print('copied', file_name)
    content = convert_pdf_to_txt(os.path.join(root, file_name))
    token = get_token(content)
    df = pd.concat([df, pd.DataFrame({"name" : file_name, "content" : content, "token" : [token]}, index=[df.shape[0]])], axis=0)
    i = df.shape[0]
    for ele in token:
        if token_id.get(ele) is None:
            token_id[ele] = {}
            token_id[ele][i] = 0
        if token_id[ele].get(i) is None:
            token_id[ele][i] = 0
        token_id[ele][i] += 1 
    print(df)
    print(token_id)
    dump_pickle('./media/uploads/data/token_id.pkl', token_id)
    dump_pickle('./media/uploads/data/id_token.pkl', id_token)
    return "pdf added"

def token_find():
    df = create_df()
    id_token = {}
    token_id = {}
    for i, content in enumerate(df.token):
        id_token[i] = content
        for ele in content:
            if token_id.get(ele) is None:
                token_id[ele] = {}
                token_id[ele][i] = 0
            if token_id[ele].get(i) is None:
                token_id[ele][i] = 0
            token_id[ele][i] += 1 
    return token_id, id_token




def find_word(word):
    token_id = load_pickle('./media/uploads/data/token_id.pkl')
    id_token = load_pickle('./media/uploads/data/id_token.pkl')
    if token_id == {}:
        token_id, id_token = token_find()
    if token_id.get(word) is None:
        return "Sorry not found"
    return token_id[word]

def find_fuzz(sent):
    print(sent)
    lst1 = {}
    similar_word = {}
    for ele in sent.split():
        if lst1.get(ele) is not None:
            continue
        a1 = {}
        for key in token_id.keys():
            fuzz_ratio = fuzz.ratio(ele, key)
            if fuzz_ratio >= 75:
                a1[key] = fuzz_ratio
                lst1[key] = [token_id[key], fuzz_ratio]
        similar_word[ele] = {k: v for k, v in sorted(a1.items(), key=lambda item: item[1])[::-1]}
    return lst1, similar_word


def find_pdf(transliteration):
    print("the transliteration is :", transliteration)
    global token_id
    global id_token
    lst1 = []
    for ele in transliteration.split():
        res = find_word(ele)
        lst1.append(res)
        print(ele,res)
    return lst1

const = "____"
def find_res(similar_word, ans):
    result = {}
    for main_ans in ans:
        score = 0
        for i, sub_ans in enumerate(main_ans):
            if i >= len(list(similar_word.keys())):
                continue
            n = list(similar_word.keys())[i]
            te_dict = similar_word[n]
            print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh", te_dict)
            if te_dict == []:
                print("yet I")
            if te_dict.get(sub_ans) is not None:
                score += te_dict.get(sub_ans)
        result[main_ans] = score
    return {k: v for k, v in sorted(result.items(), key=lambda item: item[1])[::-1]}

def find_pdf(s1):
    s2 = s1
    while True:
        s1 = " ".join(s1).strip(const).split()
        if s1 == s2:
            break
        s2 = s1
    if len(s1) == 0:
        return []
    output = {}
    print("hhhhhh", df.shape[0])
    for index in range(df.shape[0]):
        i = 0
        content = df.token[index]
        ans = []
        for ele in content:
            if s1[0] == ele:
                flag = True
                for j, ele2 in enumerate(s1):
                    if ele2 == const or i + j >= len(content): continue
                    #print("kkkkkkkkk", len(content))
                    if ele2 != content[i + j]:
                        flag = False
                        break
                if flag == True:
                    ans.append(content[i:i+len(s1)])
            i += 1
        if len(ans) != 0:
            output[index] = " ".join(ans[0])
    return output

def find_ans(str1, acc_df, similar_word):
    similar_word2 = {}
    for k, v in similar_word.items():
        for k2 in v:
            similar_word2[k2] = k
    out = pd.DataFrame()
    k = 0
    for i in range(df.shape[0]):
        res = []
        for word in str1.split():
            # print(word)
            val = []
            for ele in similar_word[word]:
                #print(ele)
                temp_df = acc_df[acc_df.word == ele].query(f"pdf_no == {i}") 
                #print(temp_df)
                if temp_df.shape[0] != 0:
                    val.append(ele)
            val.append(const)
            res.append(val)
            # print("result ", res, "\n\n")
        ans = list(itertools.product(*res))
        result = find_res(similar_word, ans)
        for j, ele in enumerate(result.keys()):
            # print(ele)
            ele2 = []
            for t in ele:
                if t != const:
                    ele2.append(similar_word2[t])
                else:ele2.append(const)
            hello = find_pdf(ele)
            if len(hello) != 0: 
                # print(j, " ".join(ele), hello, result[ele])
                for key, val in hello.items():
                    out = pd.concat([out, pd.DataFrame({"input" : " ".join(ele2), "output" : val, "pdf_no" : key, "accuracy" : result[ele], 'download' : "./media/uploads/pdf/" + list(df.iloc[[key], :].name)[0]}, index = [k])], axis=0)
                    k += 1

    if out.shape[0] != 0:
        return out.sort_values(by="accuracy", ascending=False)
    return out

def gen_res(sent, b):
    global df
    # print(df)
    if df.shape[0] == 0:
        df = create_df()
    print(df.head())
    
    temp, similar_word = find_fuzz(sent)
    acc_df = pd.DataFrame()
    i = 0
    for word, value in temp.items():
        acc = value[1]
        inside_dict = value[0]
        for pdf_no, no_of_times in inside_dict.items():
            res = acc + np.log(acc * no_of_times)
            acc_df = pd.concat([acc_df, pd.DataFrame({'word': word, 'pdf_no': pdf_no, 'res' : res, 'acc' : acc, 'no_of_times' : no_of_times, 'download' : "./media/uploads/pdf/" + list(df.iloc[[pdf_no], :].name)[0]}, index = [i])], axis=0)
            i += 1
    print(acc_df)
    if acc_df.shape[0] != 0:
        acc_df.sort_values(by="res", ascending=False)

    out = find_ans(sent, acc_df, similar_word)
    print("Thisjvb\n\n\n\n\hshgcshgcshgchgschg", out)

    if out.shape[0] == 0:
        out = pd.DataFrame({'word': "____", 'pdf_no': "not found in any pdf", 'res' : 0, 'acc' : 0, 'no_of_times' : 0, 'download' : "not available"}, index = [i])

    json_records = out.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)
    return data


def show_df():
    return df

if __name__ == "__main__":
    df = create_df()
    print(df)



# import re
# import os
# import string
# from time import time


# # import json
# import pickle
# # import glob

# # import cv2
# # from PIL import Image
# import numpy as np

# # from numpy import array
# # import pandas as pd
# # import seaborn as sns
# # import matplotlib.pyplot as plt


# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics.pairwise import cosine_similarity

# from tensorflow import keras
# # import tensorflow as tf
# # # import tensorflow.compat.v1 as tf1

# # from keras.layers.embeddings import Embedding
# # from keras.layers.merge import add

# # from tensorflow.keras import backend as K
# # from tensorflow.keras.datasets import mnist
# # from tensorflow.keras import utils
# # from keras.layers.wrappers import Bidirectional

# # from tensorflow.keras.preprocessing import image
# # from tensorflow.keras.preprocessing.text import Tokenizer
# # from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.inception_v3 import preprocess_input
# # from tensorflow.keras.utils import to_categorical
# # from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# # from tensorflow.keras.applications.vgg16 import VGG16

# from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img

# # from tqdm import tqdm



# def create_inception():
#     # model = InceptionV3(weights='imagenet')
#     # # model = VGG16(weights='imagenet')
#     # model_new = Model(model.input, model.layers[-2].output)
#     # return model_new
#     # return keras.models.load_model("C:/Users/amit chourey/Documents/computer vision/image_caption/inceptionv3_model.h5")
#     return keras.models.load_model("media/model/inceptionv3_model.h5")
# inception_model = create_inception()
# def create_real_model():
#     # return keras.models.load_model("C:/Users/amit chourey/Documents/computer vision/image_caption/model/da_model39.h5")
#     # return keras.models.load_model("media/model/da_model92.h5")
#     return keras.models.load_model("S:/All Video/MLDL/3_DJANGO/PROJECTS/data/corpus/model/hello_38.h5")
#     # return keras.models.load_model("S:/All Video/MLDL/3_DJANGO/PROJECTS/data/corpus/model/hi_1.h5")
#     # return keras.models.load_model("C:/Users/amit chourey/Documents/computer vision/img_capt/model/foot_3.h5")
#     # return keras.models.load_model("C:/Users/amit chourey/Documents/computer vision/img_capt/model/tipa_2.h5")


# real_model = create_real_model()

# def preprocess(image_path):
#     img = load_img(image_path, target_size=(299, 299))
#     x = img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     return x

# def encode(image):
#     image = preprocess(image) 
#     fea_vec = inception_model.predict(image, use_multiprocessing=True) 
#     fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
#     return fea_vec

# def load_pickle(name):
#     with open(name, "rb") as handle:
#         return pickle.load(handle)

# def greedySearch(photo):
#     max_length = 70
#     # wordtoix = load_pickle("media/model/wordtoix.pickle")
#     # ixtoword = load_pickle("media/model/ixtoword.pickle")
#     wordtoix = load_pickle("S:/All Video/MLDL/3_DJANGO/PROJECTS/data/corpus/wordtoix.pickle")
#     ixtoword = load_pickle("S:/All Video/MLDL/3_DJANGO/PROJECTS/data/corpus/ixtoword.pickle")

#     in_text = 'startseq'
#     for i in range(max_length):
#         sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
#         sequence = pad_sequences([sequence], maxlen=max_length)
#         yhat = real_model.predict([photo,sequence], verbose=0, use_multiprocessing=True)
#         yhat = np.argmax(yhat)
#         word = ixtoword[yhat]
#         in_text += ' ' + word
#         if word == 'endseq':
#             break

#     final = in_text.split()
#     final = final[1:-1]
#     final = ' '.join(final)
#     return final

# def beam_search_predictions(image, beam_index = 3):
#     max_length = 70
#     # wordtoix = load_pickle("media/model/wordtoix.pickle")
#     # ixtoword = load_pickle("media/model/ixtoword.pickle")
#     wordtoix = load_pickle("S:/All Video/MLDL/3_DJANGO/PROJECTS/data/corpus/wordtoix.pickle")
#     ixtoword = load_pickle("S:/All Video/MLDL/3_DJANGO/PROJECTS/data/corpus/ixtoword.pickle")
#     image = encode(image).reshape(1, 2048)
#     start = [wordtoix["startseq"]]
#     start_word = [[start, 0.0]]
#     while len(start_word[0][0]) < max_length:
#         temp = []
#         for s in start_word:
#             par_caps = pad_sequences([s[0]], maxlen=max_length, padding='post')
#             preds = real_model.predict([image,par_caps], verbose=0)
#             word_preds = np.argsort(preds[0])[-beam_index:]
#             # Getting the top <beam_index>(n) predictions and creating a 
#             # new list so as to put them via the model again
#             for w in word_preds:
#                 next_cap, prob = s[0][:], s[1]
#                 next_cap.append(w)
#                 prob += preds[0][w]
#                 temp.append([next_cap, prob])
                    
#         start_word = temp
#         # Sorting according to the probabilities
#         start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
#         # Getting the top words
#         start_word = start_word[-beam_index:]
    
#     start_word = start_word[-1][0]
#     intermediate_caption = [ixtoword[i] for i in start_word]
#     final_caption = []
    
#     for i in intermediate_caption:
#         if i != 'endseq':
#             final_caption.append(i)
#         else:
#             break

#     final_caption = ' '.join(final_caption[1:])
#     return final_caption

# def generate_text(url):
#     test_features = {}
#     test_features["img"] = encode(url)
#     image = test_features["img"].reshape((1,2048))
#     return greedySearch(image)

# # print(real_model.predict([np.zeros(shape=(1, 2048)), np.zeros(shape=(1, 34))], verbose=0))
# print(generate_text("media/original/images.jpg"))
# # print(beam_search_predictions("media/original/images.jpg"))

# if __name__ == '__main__':
#     print('jhjvsjs')

#     print(generate_text("media/original/images.jpg"))
#     pass



