#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import ginza
import spacy
from spacy import displacy
from pprint import pprint


# ### Analyze user input

# In[ ]:


class UserRequest():
    def __init__(self, result, entities):
        #adp = 助詞, ls = 固有表現
        self.result   = result
        self.entities = entities
        
    def get(self, keyword):
        if keyword in self.entities.keys():
            return self.entities[keyword]
        else:
            return False
        
    def get_next(self, exclude_pos=False):
        if len(self.result) > 0:
            n = self.result.pop(0)
            if exclude_pos:
                n = n[:-1]
            return "".join(n)
        else:
            return False
        
    def getall(self, exclude_pos=False):
        r = []
        n = self.get_next(exclude_pos=exclude_pos)
        while n:
            r.append(n)
            n = self.get_next(exclude_pos=exclude_pos)
        r.append(n)
        return r[:-1]

class DependencyAnalysis:
    def __init__(self):
        self.nlp = spacy.load('ja_ginza')
        self.indices = []
        self.result = []
        self.obj = []
        self.obl = []
        self.objs = []
        
    def get_analysis(self, text, display=False):
        """係り受け解析"""
        doc = self.nlp(text)
        token_head_list = []
        for sent in doc.sents:
            for token in sent:
                token_head_list.append({"i":token.i,
                                        "orth":token.orth_,
                                        "base": token.lemma_,
                                        "head": token.head.i,
                                        "dep": token.dep_,
                                        "pos":token.pos_}) 
        if display:
            pass#displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})
        return token_head_list
    
    def get_words(self,  m, head):
        """対象の領域の抽出"""
        
        if m["head"] == head:
            for n in self.depend_indices:
                self.get_words(m=n, head=m["i"])
            self.indices.append(m["i"])
        return
        
    
    def get(self, text, dep="", first=True):
        v_obj = []
        total = 0
        self.depend_indices = self.get_analysis(text)
        for sents in self.nlp(text).sents:
            target = [i for i, x in enumerate(self.depend_indices) if x["dep"] == "ROOT"][0] + total
            total += len(sents)
            head = self.depend_indices[target]["head"]
            
            for m in self.depend_indices:
                #print(m)
                if m["head"] == head and m["dep"] != "ROOT":
                    v_obj.append([m, m["dep"]])
                    
        if first:
            # 追加して など, 残りの遷移先が全て助詞の場合(右にしか遷移先がない場合)
            # rootから次の遷移先のうち、obj/oblが存在しない場合次の遷移先は名詞となる
            min_i = target
            flag = False
            for m in v_obj:
                min_i = min(min_i, m[0]["i"])
                if m[1] in ["obj", "obl", "nsubj", "acl", "advcl"]:
                    flag = True
                    
            if min_i == target:
                first = False
                
            if not flag:
                first = False
                
        if first: #これ以上深く探索する
            m = v_obj[0]
            if dep == "":
                d = m[1]
            else:
                d = dep
            self.get_words(m[0], head)
            clause = []
            for clause_num in sorted(self.indices):
                clause = clause + [str(self.depend_indices[clause_num]["orth"]).lower()]
            self.indices = []

            tmp = []
            # 1つ目の文節とそれ以外で分ける
            for x in self.depend_indices[len(clause):]:
                tmp.append(x["orth"])
            self.get("".join(clause), dep=d, first=not first)
            self.get("".join(tmp), dep="")
        else:
            r = [x["orth"].lower() for x in self.depend_indices]
            self.result.append(r)
            #もう一度解析の必要はない
            self.parse_noun("".join([x["orth"].lower() for x in self.depend_indices]), dep)

    def parse_noun(self, text, dep):
        """
        名詞の修飾関係の解析
        dep:遷移元との依存関係
        """
        depend_indices = self.get_analysis(text, display=True)
        doc = self.nlp(text)
        entities = dict(zip([x.text for x in doc.ents],
                            [x.label_ for x in doc.ents]))
        result = []
        tmp = []
        for token in doc:
            tmp.append(str(token))
            if token.pos_ == "ADP":
                result.append(tmp)
                tmp = []
        result = UserRequest(result, entities)
        if dep == "obl":
            self.obl.append(result)
        if dep == "obj":
            self.obj.append(result)
            
    def parse(self, text):
        self.result = []
        r = []
        for sents in self.nlp(text).sents: # 二文になるとうまく動作しない
            self.get(str(sents))
            r += self.result[::-1]
            self.result = []
        return r[::-1], self.obj, self.obl

