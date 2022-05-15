#!/usr/bin/env python
# coding: utf-8


import json
import pandas as pd
import siadict
import datetime
import random
from janome.tokenizer import Tokenizer
import numpy as np
from dependency_analysis import DependencyAnalysis
import os

class SIABinaryManager():
    """
    文字列の辞書データの管理
    SIA_VMコードの解析を行う
    """
    def __init__(self, bot):
        # Restore
        self.bot = bot
        self.analyzer = DependencyAnalysis()
        df = pd.read_csv("./output/sia_compiled.csv")
        df["train_Y"] = df["train_Y"].apply(lambda x: [int(a) for a in (x.replace("  ", " ").split(" "))][1:])
        
        self.compiled_codes = dict(zip(df["code"].values, df["train_Y"].values))
        df = pd.read_csv("./output/config1.csv")
        self.instructions = dict(zip(range(0, len(df["instructions"].values)),
                                     df["instructions"].values))

        self.instructions1 = dict(zip(df["instructions"].values,
                                      range(0, len(df["instructions"].values))))
        
        self.sia_dict = siadict.SIADict(len(self.instructions.keys()))
        
        df = pd.read_csv("./output/config2.csv")
        k = len(self.sia_dict.sia_encode.keys()) + 1 + len(self.instructions.keys())
        
        self.pronouns = dict(zip(range(k, k+len(df["pronouns"].values)), df["pronouns"].values))
        self.pronouns1 = dict(zip(df["pronouns"].values, range(k, k+len(df["pronouns"].values))))
        self.tokenize = Tokenizer().tokenize
        
    def parse_string(self, strings):
        tmp = []
        result = []
        for string_int in strings:
            if string_int == self.plus:
                result.append(tmp)
                tmp = []
            else:
                tmp.append(string_int)
                
        if len(tmp) > 0:
            result.append(tmp)
        return result
        
    def interpret(self, sia_vm_code):
        result = []
        tmp = []
        for c in sia_vm_code:
            if c in self.instructions.keys() and len(tmp) == 0:
                tmp.append(self.instructions[c])
            elif c in self.instructions.keys() and len(tmp) != 0:
                result.append(tmp)
                tmp = []
                tmp.append(self.instructions[c])
            elif c in self.pronouns.keys():
                tmp.append(self.pronouns[c])
            else:
                tmp.append(c)
        if len(tmp) > 0:
            result.append(tmp)
        return result
    
    def parse_sentence_tree(self, sentence):
        tokens = [x for x in self.tokenize(sentence)]
        result = {"名詞":[], "助詞":[], "形容詞":[], "副詞":[], "動詞":[]}
        for i in range(0, len(tokens) -1):
            k = tokens[i+1].part_of_speech.split(",")[0]
            k1 = tokens[i+1].part_of_speech.split(",")[1]
            if k in result.keys() and not k1 in ["接続助詞", "終助詞"]:
                result[k].append(tokens[i].surface)
        return result
    
    def total_dict_len(self):
        return len(self.sia_dict.sia_encode.keys()) + len(self.pronouns.keys()) + len(self.instructions.keys())
    
    def call_code(self, code):
        return self.compiled_codes[code]
    
    def decode_str(self, ints):
        return "".join(self.sia_dict.sia_text_decode(ints))
    
    def encode_word(self, word):
        return self.sia_dict.sia_encode_word(word)
    
    def encode_s(self, sentence):
        return self.sia_dict.sia_text_encode(sentence)

# In[ ]:
class UserInfoData():
    """
    ユーザー情報 (住んでる場所, プレイリスト, 予定...)を管理
    """
    def __init__(self, client_id):
        config_file_path = "./client_data/{}.json".format(str(client_id))
        self.config_file_path = config_file_path
        if os.path.exists(config_file_path):
            f = open(config_file_path, "r")
            self.config = json.load(f)
            f.close()
        else:
            print("New client_data file created at ./client_data/{}.json".format(str(client_id)))
            self.config = {}
            with open(config_file_path, "w") as f:
                f.write(json.dumps(self.config))
                
        self.memo_prefix = "メモ_"
                
    def save_current_state(self):
        with open(self.config_file_path, "w") as f:
            f.write(json.dumps(self.config))
            
    def get_target(self, target):
        """
        メモ/予定/プレイリストなどの種類の管理
        """
        if target in self.config.keys():
            return self.config[target]
        else:
            self.config[target] = {}
            self.save_current_state()
            return self.config[target]
        
    def write_dict(self, target, key, content):
        target = self.get_target(target)
        if key in target.keys():
            target[key].append(content)
        else:
            target[key] = [content]
        self.save_current_state()
        
    def delete_dict(self, target, key):
        target = self.get_target(target)
        if key in target.keys():
            target[key] = []
        self.save_current_state()
        
    def get_memos(self, key):
        target = self.get_target("memo")
        result = []
        for k in target.keys():
            if key in k:
                result.append(k)
                
        memos = ""
        for r in result:
            title = "[{}]\n".format(r)
            memos += title
            for i, memo in enumerate(target[r]):
                memos += "{}. {}\n".format(str(i), memo)
            
        return memos

class SIAInstruction():
    def __init__(self):
        with open("./config/respond.json", "r") as f:
            self.respond = json.load(f)
        with open("./commands/simple_help.txt", "r") as f:
            self.simple_help = f.read()
        with open("./commands/introduction.txt", "r") as f:
            self.introduction = f.read()
        with open("./commands/show_examples.txt", "r") as f:
            self.show_examples = f.read() # instructions.txtからランダムで
        with open("./commands/help.txt", "r") as f:
            self.detailed_help = f.read()
            
    async def execute(self, session, ctx, manager, args):
        pass
    
    def get_pronouns(self, session, ctx, pronouns):
        """
        Push System Variable
        """
        if pronouns == "your_name":
            return ctx.author.name
        elif pronouns == "greeting":
            date = datetime.datetime.now().hour
            hours = np.arange(1,25)
            if date in hours[7:11]:
                result = self.respond["greeting-morning"]
            elif date in hours[12:16]:
                result = self.respond["greeting-daily"]
            elif date in hours[17:21]:
                result = self.respond["greeting-night1"]
            elif date in hours[21:-1]:
                result = self.respond["greeting-night2"]
            elif date in hours[0:3]:
                result = self.respond["greeting-night3"]
            else:
                result = self.respond["greeting-night4"]
                
            random.shuffle(result)
            return result[0]
        elif pronouns == "助詞":
            if len(session.args["助詞"]) == 0:
                #r = self.user_input_request(ctx, "何について調べる？", "遅い。。。もう一回やって。。。") # it won't works...
                return "nothing"
            else:
                r = session.args["助詞"][0]
                session.args["助詞"] = session.args["助詞"][1:]
                return r
        elif pronouns == "simple_sia_help":
            return self.simple_help
        elif pronouns == "show_examples":
            return self.show_examples
        elif pronouns == "show_detailed_help":
            return self.detailed_help
        elif pronouns == "introduction":
            return self.introduction
        else:
            return "[Undefined pronouns : {}]".format(pronouns)

from commands import instructions

class SIAVMSession():
    """
    SIAVM
    """
    def __init__(self, client_id):
        self.stack = []
        self.execution_code = []
        self.ep = 0
        self.obj = ""
        self.obl = ""
        self.args = []
        self.user_info = UserInfoData(client_id)
        self.instructions = []
        self.user_last_input = ""
        self.commands = {}
        
        with open("./commands/instructions.json", "r") as f:
            config = json.load(f)
            
        for opecode in config["instructions"]: #Manager側で実装したほうがいいかも。。。
            self.commands[opecode] = getattr(instructions, opecode)()
        
    def stack_push(self, obj):
        self.stack.append(obj)
    
    def stack_pop(self):
        return self.stack.pop()
    
    async def send(self, ctx, manager, name, args):
        return await self.commands[name].execute(self, ctx, manager, args)
        
    def set_vm(self, manager, sia_vm_code):
        self.instructions = manager.interpret(sia_vm_code)
        self.ep = 0
        
    def display_i(self):
        return self.instructions
        
    async def vm_next(self, ctx, manager):
        code = self.instructions[self.ep]
        self.ep += 1
        await self.send(ctx, manager, code[0], code[1:])
    
    async def vm_run(self, ctx, manager):
        m = len(self.instructions)
        
        while self.ep < m:
            await self.vm_next(ctx, manager)
    
    def set_code(self, manager, code):
        vmcode = manager.call_code(code)
        return self.set_vm(manager, vmcode)
    
    def parse_args(self, manager, sentence):
        self.user_last_input = sentence

