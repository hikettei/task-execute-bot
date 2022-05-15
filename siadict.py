#!/usr/bin/env python
# coding: utf-8

# ### 辞書を作る際に集めるファイルたち
# 
# input/instruction_txt.txt
# 
# input/instruction/にあるreply

# In[ ]:


import pandas as pd
from janome.tokenizer import Tokenizer
from tqdm import tqdm
import json
import numpy as np


# In[ ]:


class SIADict():
    def __init__(self, instruction_space, make_dict=False, ignore_json=True):
        self.tokenize = Tokenizer().tokenize
        self.input_files = ["./train/discord-dialogs.json"]
        
        if ignore_json:
            self.input_files = []
            
        if make_dict:
            self.make_dict()
        vocab = pd.read_csv("vocab.csv")
        instruction_space += 1
        k = instruction_space + len(vocab["vocab"])
        self.sia_encode = dict(zip(vocab["vocab"].values, range(instruction_space, k)))
        self.sia_decode = dict(zip(range(instruction_space, k),vocab["vocab"].values))
        
    def sia_encode_word(self, word):
        if word in self.sia_encode.keys():
            return self.sia_encode[word]
        else:
            return self.sia_encode["<UNK>"]

    def sia_text_encode(self, text):
        return [self.sia_encode_word(x.surface) for x in self.tokenize(text)]

    def sia_text_decode(self, ints):
        return [self.sia_decode[x] for x in ints]
    
    def str_filter(self, program_line):
        tokens = program_line.split(" ")
        result = []
        for token_list in tokens:
            for token in token_list.split("+"):
                if token.startswith("\"") and token.endswith("\""):
                    result.append(token[1:][:-1]) # strings
        return result
    
    def collect_json_words(self):
        result = []
        print("Processing for {} extend files...".format(len(self.input_files)))
        print("")
        for file_path in self.input_files:
            f = open(file_path, "r")
            log = json.load(f)
            for sentence in tqdm(log):
                result += list(np.unique([x.surface for x in self.tokenize(sentence)]))
        return list(np.unique(result))
    
    def make_dict(self):
        tokenize = Tokenizer().tokenize
        instruction_txt = pd.read_csv("input/instruction_txt.txt")
        inflated_df = pd.read_csv("input/inflated_data.csv")
        instruction_txt = pd.concat([instruction_txt, inflated_df])
        
        sia_words = []
        print("Collecting words...")
        print("")
        for file_path, input_txt in tqdm(zip(instruction_txt["file_path"].values,
                                             instruction_txt["input_txt"].values), total=len(instruction_txt["file_path"].values)):
            tokens = [x.surface for x in tokenize(input_txt)]
            df = pd.read_csv("./input/instructions/" + file_path + ".sia")
            strs =  self.str_filter(" ".join(df["SIA"].values))
            for t in strs:
                tokens += [x.surface for x in tokenize(t)]
            sia_words += list(np.unique(tokens))
        sia_words += ["<EOS>"]
        sia_words += ["<UNK>"]
        sia_words += ["<BEGIN>"]
        sia_words += [" "]
        sia_words += self.collect_json_words()
        sia_words = list(np.unique(sia_words))
        df = pd.DataFrame(sia_words, columns=["vocab"])
        df.to_csv("vocab.csv")

