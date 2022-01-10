import math
from inverted_index_gcp import *
import numpy as np
import pandas as pd


class BM25_from_index:

    def __init__(self,index,base_dir='',k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.DL)
        self.AVGDL = sum(index.DL.values())/self.N
        self.base_dir = base_dir

    def get_candidate_documents(self,query_to_search,base_dir):
        candidates = []
        for term in np.unique(query_to_search):
            if term in self.index.df.keys():
                current_list = self.index.read_posting_list(term,base_dir)
                candidates += current_list
        candidates = set([x[0] for x in candidates])
        return set(candidates)

    def calc_idf(self,list_of_tokens):
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def _score(self,query, doc_id, pls_dict):
        score = 0.0
        doc_len = self.index.DL[doc_id]
        for term in query:
            if term in self.index.df.keys():
                if doc_id in pls_dict[term].keys():
                    freq = pls_dict[term][doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score


    def search(self, query,N=100):
        idf = self.calc_idf(query)
        self.idf = idf
        pls_dict = {}
        for term in query:
            if term in self.index.df.keys() and term not in pls_dict:
                pls_dict[term] = dict(self.index.read_posting_list(term ,self.base_dir))
        candidates = self.get_candidate_documents(query,self.base_dir)
        lst_B25 = [(c, self._score(query,c,pls_dict)) for c in candidates]
        lst_B25 = sorted(lst_B25 , key = lambda x: x[1], reverse = True)[:N]
        return lst_B25
