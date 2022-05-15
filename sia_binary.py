#!/usr/bin/env python
# coding: utf-8

# ### SIA Binary
# 
# ### Operands
# 
# 0 ... exit_session
# 1 ... reply msg
# 2 ... delete_message
# 3 ... push_message
# 
# ### 形式
# .siaファイルは、一行目がSIAで始まる。
# 
# 命令ごとに改行をしていく
# 
# 文字列の中で空白の使用は認められない。
# 
# ダブルクオーテーションで囲ったらそれは文字列、括っていなかったらそれは変数名として扱われる
# 
# ### 数字の割り当て
# 
# 最初の方... instruction_list
# 中盤   ... string
# 最後   ... pronouns sia_dict_size + len(pronoun_dict.keys()) + 1 + len(instruction_list)

# In[ ]:


import pandas as pd
import re
from tqdm import tqdm
import siadict
import numpy as np
import argparse
from janome.tokenizer import Tokenizer
from gensim.models import word2vec
import random
import json
import gc


# In[ ]:


with open("./commands/instructions.json", "r") as f:
    instruction_config = json.load(f)


# In[ ]:


instruction_list = []

for instruction in instruction_config["instructions"]:
    instruction_list.append(instruction)

instruction2int = dict(zip(instruction_list, range(0,len(instruction_list))))
int2instruction = dict(zip(range(0,len(instruction_list)), instruction_list))


# In[ ]:


instruction_txt = pd.read_csv("input/instruction_txt.txt")
sia_dict_size = 0
sia_dict = 0

pronoun_dict = {}

def get_pronoun(pr):
    if pr in pronoun_dict.keys():
        return pronoun_dict[pr]
    else:
        pronoun_dict[pr] = sia_dict_size + len(pronoun_dict.keys()) + 1 + len(instruction_list)
        return pronoun_dict[pr]


# # The method to prepare sufficient datas
# 
# ## Methods
# 
# ### log -> Inflating total data by dialogs.json
# 
# ### sim -> Generating similar texts
# 
# ### nope -> do nothing
# 
# ## Params
# 
# file_path ... When sia received input_txt, she runs file_path.sia
# 
# input_txt ... input_txt.
# 
# method ... the way to inflate input_txt, show methods.
# 
# rate ...   the number of inflating is determined by k*rate, which k is command line parameter.

# In[ ]:


w2v_model = word2vec.Word2Vec.load("./config/w2v.model").wv
f = open("train/dialogs.json", 'r')
dialogs = json.load(f)
tokenizer = Tokenizer()

def shuffle_log(log,n=100):
    log1 = [log[i:i+n] for i in range(0, len(log), n)]
    indices = np.arange(0,len(log1))
    random.shuffle(indices)
    res = []
    for i in indices:
        res += log1[i]
    return res

def sentence_similarity(x,y):
    tokensx = tokenizer.tokenize(x)
    tokensy = tokenizer.tokenize(y)
    score = 0.
    count = 0.
    for tokenx in tokensx:
        for tokeny in tokensy:
            if len(tokenx.surface) >= 2 and len(tokeny.surface) >= 2:
                try:
                    score+=w2v_model.similarity(tokenx.surface,tokeny.surface)
                    count+=1.
                except KeyError:
                    pass

    if score == 0.:
        return 0.
    return (score/count)

def find_similar_sentences(trigger, k, topn=20000):
    res = {}
    random.shuffle(dialogs)
    for s in dialogs[0:topn]:
        score = sentence_similarity(trigger, s)
        if score >= 0.51:
            res[s] = score
            
        if len(res.keys()) >= (k + 10):
            break
            
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())
    return res[:k]

def generate_similar_sentences(trigger, k):
    tokens = [x for x in tokenizer.tokenize(trigger)]
    ignores = []
    
    for word in tokens:
        try:
            w2v_model[word.surface]
        except KeyError:
            ignores.append(word.surface)
            
    tmp = []
    result = {"名詞":[], "助詞":[], "形容詞":[], "副詞":[], "動詞":[]}
    keys = ["名詞", "動詞"]
    
    for i in range(0, len(tokens)):
        ks = tokens[i].part_of_speech.split(",")[0]
        if ks in result.keys():
            if not tokens[i].surface in ignores:
                result[ks].append(tokens[i].surface)
                
    for _ in range(k+3):
        input_tmp = trigger
        for i in range(random.randint(1, 2)):
            random.shuffle(keys)
            key = keys[0]
            if len(result[key]) > 0:
                random.shuffle(result[key])
                word = result[key][-1]
                try:
                    similar_words = w2v_model.most_similar(positive=word, topn=2)
                    i = random.randint(0,1)
                    input_tmp = input_tmp.replace(word, similar_words[i][0])
                except KeyError:
                    pass
        tmp.append(input_tmp)
    tmp = list(np.unique(tmp))
    return tmp[:k]

def reshape_data(file_path, sentences):
    return [[file_path, sentence, "D", 0] for sentence in sentences]

def bump_sia_file(reply):
    return "\n".join(["SIA", "push \"" + reply.replace(" ", "") + "\"", "reply", "send sia_talk_api"])

def inflate_data(k):
    inflated_data = []
    
    a_data_num = 1
    b_data_num = 1
    
    a_inflated_num = 0
    b_inflated_num = 0
    
    print("Inflating data...")
    print("")
    
    for file_path, input_txt, method, rate in tqdm(zip(instruction_txt["file_path"].values,
                                                       instruction_txt["input_txt"].values,
                                                       instruction_txt["method"].values,
                                                       instruction_txt["rate"].values), total=len(instruction_txt["file_path"].values)):
        k = round(k*int(rate))
        method = method.replace(" ", "")
        if k <= 0:
            k = 0
        if method == "A":
            a_data_num += 1
            sentence_list = find_similar_sentences(input_txt, k)
            a_inflated_num += len(sentence_list)
            inflated_data += reshape_data(file_path, sentence_list)
        elif method == "B":
            b_data_num += 1
            sentence_list = generate_similar_sentences(input_txt, k)
            b_inflated_num += len(sentence_list)
            inflated_data += reshape_data(file_path, sentence_list)
        elif method == "C":
            # Do nothing
            pass
        elif method == "D":
            raise RuntimeError("You're trying to inflate inflated-dataset")
        else:
            raise RuntimeError("Found syntax error in instruction_txt.txt")
            
    total = np.sum([a_data_num, b_data_num, a_inflated_num, b_inflated_num])
    print("=Inflating finished:==============================")
    # Result
    print("Inflated by similar_sentences  : Total ... from {} to {} Expand rate : {}".format(a_data_num, a_data_num + a_inflated_num, a_inflated_num/a_data_num))
    
    print("Inflated by generating_sentence: Total ... from {} to {} Expand rate : {}".format(b_data_num, b_data_num + b_inflated_num, b_inflated_num/b_data_num))
    
    print("The total amount of data is {}".format(total))
    print("")
    
    return pd.DataFrame(inflated_data, columns=["file_path", "input_txt", "method", "rate"])


# In[ ]:


def compile_binary(program_line):
    tokens = program_line.split(" ")
    result = []
    for i, token_list in enumerate(tokens):
        if not token_list == "":
            for token in token_list.split("+"):
                if token in instruction_list:
                    result.append(instruction2int[token])
                elif token.startswith("\"") and token.endswith("\""):
                    result += sia_dict.sia_text_encode(token)[1:][:-1] # strings
                else:
                    result.append(get_pronoun(token)) # variable
    return " " + str(result)[1:][:-1].replace(",", " ")
    
def compile_from_source(file_paths):
    result = ""
    for file_path in file_paths.split("+"):
        file_path = file_path.replace(" ", "")
        df = pd.read_csv("./input/instructions/" + file_path + ".sia")
    
        try:
            df["SIA"]
        except KeyError:
            print("Format Warning: {file_path} is not SIA Program, Ignored".format(file_path))
            return False
    
        result += compile_binary(" ".join(df["SIA"].values))
    return (str(sia_dict.sia_encode_word("<BEGIN>")) + result + " " + str(sia_dict.sia_encode_word("<EOS>"))).replace("  ", " ")


# In[ ]:


def start_compile(args):
    global instruction_txt
    global sia_dict_size
    global sia_dict
    if args.checkpoint == "True":
        df = inflate_data(int(args.k))
        df.to_csv("./input/inflated_data.csv")
        instruction_txt = pd.concat([instruction_txt, df], axis=0).reset_index()
    else:
        df = pd.read_csv("./input/inflated_data.csv")
        instruction_txt = pd.concat([instruction_txt, df], axis=0).reset_index()
        
    if args.ignore_json == "True":
        ignore_json = True
    else:
        ignore_json = False
    
    sia_dict = siadict.SIADict(len(instruction_list), make_dict=True, ignore_json=ignore_json)
    sia_dict_size = len(sia_dict.sia_encode.keys())
    
    train = []
    print("Collecting Training File...")
    print("")
    for file_path, input_txt in tqdm(zip(instruction_txt["file_path"].values,
                                         instruction_txt["input_txt"].values), total=len(instruction_txt["file_path"].values)):
        train.append([input_txt, compile_from_source(file_path), file_path])
        
    train_data_df = pd.DataFrame(train, columns=["train_X", "train_Y", "code"])
    config_df1 = pd.DataFrame(instruction_list, columns=["instructions"])
    config_df2 = pd.DataFrame(list(pronoun_dict.keys()), columns=["pronouns"])
    
    train_data_df.to_csv("./output/sia_compiled.csv")
    config_df1.to_csv("./output/config1.csv")
    config_df2.to_csv("./output/config2.csv")
    print("Saved output file.")
    print("")

