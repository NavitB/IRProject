import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import math
from operator import add
import builtins
from google.cloud import storage
from collections import defaultdict
from inverted_index_gcp import *
import hashlib
import math
from itertools import chain
import time
from BM25 import BM25_from_index
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

# nltk.download('stopwords')

# import pyspark
# from pyspark.sql import *
# from pyspark.sql.functions import *
# from pyspark import SparkContext, SparkConf, SparkFiles
# from pyspark.sql import SQLContext
# from pyspark.ml.feature import Tokenizer, RegexTokenizer
# from graphframes import *

# sc.addFile("/home/dataproc/inverted_index_gcp.py")
# sys.path.insert(0,SparkFiles.getRootDirectory())


class myBackEnd():

    def __init__(self):
        self.DL_title = {}
        self.vec_len_doc_body = {}
        self.term_total_body = {}
        self.DL_body = {}
        bucket_name = "318457645"
        client = storage.Client()
        blobs = client.list_blobs(bucket_name)
        for blob in blobs:
            if(blob.name == 'postings_gcp/index.pkl'):
                with blob.open("rb") as f:
                    self.index_body = pickle.load(f)
            if(blob.name == 'postings_gcp/text_DL.pickle'):
                with blob.open("rb") as f:
                    self.DL_body = pickle.load(f)
            if(blob.name == 'postings_gcp/vec_len_total.pickle'):
                with blob.open("rb") as f:
                    self.vec_len_doc_body = pickle.load(f)
            if(blob.name == 'postings_gcp/text_term_total.pickle'):
                with blob.open("rb") as f:
                    self.term_total_body = pickle.load(f)
            if(blob.name == 'postings_gcptitle/index_title.pkl'):
                with blob.open("rb") as f:
                    self.index_title = pickle.load(f)
            if(blob.name == 'postings_gcptitle/DL_title.pickle'):
                with blob.open("rb") as f:
                    self.DL_title = pickle.load(f)
            if(blob.name == 'pv/pageviews-202108-user.pkl'):
                with blob.open("rb") as f:
                    self.wid2pv = pickle.load(f)
            if(blob.name == 'postings_gcpanchor/index_anchor.pkl'):
                with blob.open("rb") as f:
                    self.index_anchor = pickle.load(f)
            if(blob.name == 'postings_gcptitle/id_title_dict.pickle'):
                with blob.open("rb") as f:
                    self.id_title_dict = pickle.load(f)
            if(blob.name == 'pr/pr_part-00000-c1f636d8-972a-483c-a516-d1b60f8e868f-c000.csv'):

                content = blob.download_as_string()
                df = pd.read_csv(BytesIO(content))
                df.columns = ["id","pageRank"]
                self.page_rank = df


        self.index_body.DL = self.DL_body
        self.index_body.vec_len_doc = self.vec_len_doc_body
        self.index_body.term_total = self.term_total_body
        self.index_title.DL = self.DL_title

#############reading and writing funtions ##################
    # def open_gcp(file_name,dir_name):
    #     client = storage.Client(file_name)
    #     bucket = client.bucket(bucket_name)
    #     blob = bucket.get_blob(f'postings_gcp{dir_name}/' + file_name)
    #     return blob.open('rb')
    #
    # def read_pickle(file_name,dir_name):
    #     stream = open_gcp(file_name+".pickle",dir_name)
    #     pick = pickle.load(stream)
    #     stream.close()
    #     return pick
    #
    # def read_pkl(file_name,dir_name):
    #     stream = open_gcp(file_name+".pkl",dir_name)
    #     pick = pickle.load(stream)
    #     stream.close()
    #     return pick

###############start body search###################

    # def read_all_files_body(self):
    #     DL_dict = read_pickle('text_DL','')
    #     self.index_body.DL = DL_dict
    #     term_total_dict = read_pickle('text_term_total','')
    #     self.index_body.term_total = term_total_dict
    #     vec_len_dict = read_pickle('vec_len_total', '')
    #     self.index_body.vec_len_doc = vec_len_dict

    def generate_query_tfidf(self,query_to_search,index):
        dict_term_tfidf = {}
        counter = Counter(query_to_search)
        for term in query_to_search:
            # assert (term in index.df.keys() , "problem with df" + str(index.df[term]))
            if term in index.df.keys(): #avoid terms that do not appear in the index.
                tf = counter[term]/len(query_to_search) # term frequency divded by the length of the query
                df = index.df[term]
                idf = math.log((len(index.DL))/(df),10) #smoothing
                dict_term_tfidf[term] = tf*idf
        return dict_term_tfidf

    def get_candidate_documents_and_scores(self,query_to_search,index,base_dir=''):
        candidates = {}
        N = len(index.DL)
        for term in np.unique(query_to_search):
            if term in index.df.keys():
                list_of_doc = index.read_posting_list(term, base_dir)
                normlized_tfidf = [(doc_id,(freq/index.DL[doc_id])*math.log(N/index.df[term],10)) for doc_id, freq in list_of_doc]
                for doc_id, tfidf in normlized_tfidf:
                    candidates[(doc_id,term)] = candidates.get((doc_id,term), 0) + tfidf
        return candidates

    def cosine_similarity(self,tfidf_doc,dict_query,index,doc_id):
        #for one query and doc
        cos_sim_mone = 0.0
        query_size_vec = 0.0
        for term in dict_query.keys():
            cos_sim_mone += dict_query[term] * tfidf_doc
            query_size_vec += math.pow(dict_query[term],2)
        cos_sim_total = cos_sim_mone / (index.vec_len_doc[doc_id] * (math.sqrt(query_size_vec)))
        return cos_sim_total

    def generate_cosine_dict(self,query_to_search,index,base_dir):
        cosine_dict = {}
        candidates_scores = self.get_candidate_documents_and_scores(query_to_search,index,base_dir) #We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
        dict_query = self.generate_query_tfidf(query_to_search,index)
        for key in candidates_scores:
            doc_id, term = key
            cosine_dict[doc_id] = self.cosine_similarity(candidates_scores[key],dict_query,index,doc_id)
        return cosine_dict

    def get_top_n(self,query,index, base_dir='',N=100):
        sim_dict = self.generate_cosine_dict(query,index, base_dir)
        lst = [(doc_id , builtins.round(score,5)) for doc_id, score in sim_dict.items()]
        lst = sorted(lst, key = lambda x: x[1], reverse=True)[:N]
        lst = [(x[0],self.id_title_dict[x[0]]) for x in lst]
        return lst

    def body_search(self,query):
        # self.read_all_files_body()
        return self.get_top_n(query.split(' '), self.index_body)



###############end body search###################

###############start title search###################


    def get_score_for_title(self,query, index):
        docs_count = {}
        all_relevant_docs = []
        len_query = len(np.unique(query))
        for term in np.unique(query):
            if term in index.df.keys():
                list_of_doc = index.read_posting_list(term, 'title')
                list_of_doc = [x[0] for x in list_of_doc]
                all_relevant_docs.extend(list_of_doc)
        set_all = set(all_relevant_docs)
        for id in set_all:
            if id not in docs_count.keys():
                docs_count[id] = all_relevant_docs.count(id)
        docs_count = sorted(docs_count.items(), key=lambda x: x[1], reverse=True)
        docs_count = [(x[0],self.id_title_dict[x[0]]) for x in docs_count]
        return docs_count


    def title_search(self,query):
        # self.read_files_title()
        return self.get_score_for_title(query.split(' '), self.index_title)


###############end title search###################

###############start anchor search###################


    def get_score_for_anchor(self,query, index):
        docs_count = {}
        all_relevant_docs = []
        len_query = len(np.unique(query))
        for term in np.unique(query):
            if term in index.df.keys():
                list_of_doc = index.read_posting_list(term, 'anchor')
                list_of_doc = [x[0] for x in list_of_doc]
                all_relevant_docs.extend(list_of_doc)
        set_all = set(all_relevant_docs)
        for id in set_all:
            if id not in docs_count.keys():
                docs_count[id] = all_relevant_docs.count(id)
        docs_count = sorted(docs_count.items(), key=lambda x: x[1], reverse=True)
        docs_count = [(x[0],self.id_title_dict[x[0]]) for x in docs_count]
        return docs_count

    def anchor_search(self,query):
        # self.read_files_title()
        return self.get_score_for_anchor(query.split(' '), self.index_anchor)


###############end anchor search###################

###############start main search###################


    def merge_results(self,title_scores,body_scores,title_weight=0.5,text_weight=0.5,N = 100):
        # dict_topN = {}
        # for q_id in title_scores:
        tuples_title = title_scores
        doc_id_title = [x[0] for x in tuples_title]

        tuples_body = body_scores
        doc_id_body = [x[0] for x in tuples_body]

        list_diff_id = [x for x in doc_id_title if x not in doc_id_body] + [x for x in doc_id_body if x not in doc_id_title]
        list_same_id = [x for x in doc_id_title if x not in list_diff_id]

        tuples_diff = [(x[0], x[1] * title_weight) for x in tuples_title if x[0] in list_diff_id] + [(x[0], x[1] * text_weight) for x in tuples_body if x[0] in list_diff_id]

        dict_same = {}
        for x, y in tuples_title:
            if(x in list_same_id):
                dict_same.setdefault(x, []).append(y)
        for x, y in tuples_body:
            if(x in list_same_id):
                  dict_same.setdefault(x, []).append(y)
        tuples_same = [(x, y[0] * title_weight + y[1] * text_weight) for x, y in dict_same.items()]

        new_tuples = tuples_same + tuples_diff
        new_tuples = sorted(new_tuples, key=lambda t: t[1], reverse=True)[:N]
        lst = [(x[0],self.id_title_dict[x[0]]) for x in new_tuples]
        return lst

    def helper_search(self,query):
        bm25_title = BM25_from_index(self.index_title,'title')
        bm25_text = BM25_from_index(self.index_body,'')
        bm25_title_results = bm25_title.search(query,100)
        bm25_body_results = bm25_text.search(query,100)
        merged = self.merge_results(bm25_title_results,bm25_body_results)
        return merged

    def main_search(self,query):
        return self.helper_search(query.split(' '))
