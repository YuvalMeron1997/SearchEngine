import numpy as np
from inverted_index_gcp import InvertedIndex
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import math
from nltk.corpus import stopwords
# from pyspark import SparkContext
import re
import concurrent.futures
import pandas as pd
import threading

import nltk

nltk.download('stopwords')


class SearchBackend:
    def __init__(self):
        bucket_name = '209044148'
        self.body_index = InvertedIndex.read_index('postings_gcp', 'index', bucket_name)
        self.title_index = InvertedIndex.read_index('index_title', 'index_t', bucket_name)
        self.anchor_index = InvertedIndex.read_index('index_anchor', 'index_a', bucket_name)
        self.dl = InvertedIndex.read_index('', 'dl', bucket_name)
        self.tl = InvertedIndex.read_index('', 'tl', bucket_name)
        self.al = InvertedIndex.read_index('', 'al', bucket_name)
        self.idf_body = InvertedIndex.read_index('', 'idf', bucket_name)
        self.idf_title = InvertedIndex.read_index('', 'idf_title', bucket_name)
        self.idf_anchor = InvertedIndex.read_index('', 'idf_anchor', bucket_name)
        self.norm_body = InvertedIndex.read_index('', 'norm_doc', bucket_name)
        self.norm_title = InvertedIndex.read_index('', 'norm_title', bucket_name)
        self.norm_anchor = InvertedIndex.read_index('', 'norm_anchor', bucket_name)
        self.doc_title_pairs = InvertedIndex.read_index('', 'doc_title_pairs', bucket_name)
        # part-00000-ded794d0-5d7c-4e49-9068-c1b2ce6ca215-c000.csv.gz
        self.page_rank = InvertedIndex.read_index('', 'pr', bucket_name)

    # new
    def search_body(self, q):
        res = []
        query = q
        if len(query) == 0:
            return jsonify(res)

        # Tokenize the query
        query_terms = query.split()
        query_length = len(query_terms)

        q_weights = {}
        for term in query_terms:
            if term in self.body_index.df.keys():
                tf = query_terms.count(term) / query_length
                q_weights[term] = tf * self.idf_body[term]
            else:
                continue

        doc_ranks = {}
        doc_weight = {}
        lock = threading.Lock()

        def process_term(term):
            if self.idf_body[term] >= 1:
                nonlocal doc_ranks, doc_weight
                posting_list = self.body_index.read_a_posting_list(base_dir='postings_gcp', w=term,
                                                                   bucket_name='209044148')
                for doc_id, tf in posting_list:
                    weight = (tf / self.dl[doc_id]) * self.idf_body[term]
                    with lock:
                        doc_weight[(doc_id, term)] = weight
                        doc_ranks[doc_id] = doc_ranks.get(doc_id, 0) + q_weights[term] * weight


        with concurrent.futures.ThreadPoolExecutor() as executor:
            for key in q_weights.keys():
                executor.submit(process_term, key)

        # Calculate normalization constants
        counter_q_norm = sum(q ** 2 for q in q_weights.values())
        counter_q_norm = 1 / math.sqrt(counter_q_norm) if counter_q_norm != 0 else 0

        # Normalization
        for doc_id in doc_ranks.keys():
            doc_ranks[doc_id] *= counter_q_norm * 1 / self.norm_body[doc_id]

        sorted_docs = sorted(doc_ranks.items(), key=lambda x: x[1], reverse=True)[:100]
        res.extend(sorted_docs)

        return res


    def title_search(self, q):
        res = []
        query = q
        if len(query) == 0:
            return jsonify(res)

        # Tokenize the query
        query_terms = query.split()
        query_length = len(query_terms)

        q_weights = {}
        for term in query_terms:
            if term in self.title_index.df.keys():
                tf = query_terms.count(term) / query_length
                q_weights[term] = tf * self.idf_title[term]
            else:
                continue

        title_ranks = {}
        title_weight = {}
        lock = threading.Lock()

        def process_term(term):
            nonlocal title_ranks, title_weight
            if self.idf_title[term] >= 1.5:
                posting_list = self.title_index.read_a_posting_list(base_dir='', w=term, bucket_name='209044148')
                for doc_id, tf in posting_list:
                    weight = (tf / self.tl[doc_id]) * self.idf_title[term]
                    with lock:
                        title_weight[(doc_id, term)] = weight
                        title_ranks[doc_id] = title_ranks.get(doc_id, 0) + q_weights[term] * weight

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for key in q_weights.keys():
                executor.submit(process_term, key)

        # Calculate normalization constants
        counter_q_norm = sum(q ** 2 for q in q_weights.values())
        counter_q_norm = 1 / math.sqrt(counter_q_norm) if counter_q_norm != 0 else 0

        # Normalization
        for doc_id in title_ranks.keys():
            title_ranks[doc_id] *= counter_q_norm * 1 / self.norm_title[doc_id]

        sorted_docs = sorted(title_ranks.items(), key=lambda x: x[1], reverse=True)[:100]
        res.extend(sorted_docs)

        return res

    # def search_anchor(self,q):
    #     res = []
    #     # query = request.args.get('query', '')
    #     query = q
    #     if len(query) == 0:
    #         return jsonify(res)
    #     # BEGIN SOLUTION
    #
    #     # Tokenize the query
    #     query_terms = query.split()
    #     query_length = len(query_terms)
    #
    #     q_weights = {}
    #     for term in query_terms:
    #         if term in self.anchor_index.df.keys():
    #             tf = query_terms.count(term) / query_length
    #             q_weights[term] = tf * self.idf_anchor[term]
    #         else:
    #             continue
    #
    #     doc_ranks = {}
    #     doc_weight = {}
    #     for term in q_weights.keys():
    #         posting_list = self.anchor_index.read_a_posting_list(base_dir='', w=term, bucket_name='209044148')
    #         for doc_id, tf in posting_list:
    #             doc_weight[(doc_id, term)] = (tf / self.al[doc_id]) * self.idf_anchor[term]
    #             doc_ranks[doc_id] = 0
    #
    #     for doc_id in doc_ranks.keys():
    #         r = 0
    #         for term in q_weights.keys():
    #             if (doc_id, term) in doc_weight:
    #                 r += q_weights[term] * doc_weight[(doc_id, term)]
    #
    #         doc_ranks[doc_id] = r
    #
    #     # constant
    #     counter_q_norm = 0
    #     for q in q_weights.values():
    #         counter_q_norm += q ** 2
    #     counter_q_norm = math.sqrt(counter_q_norm)
    #     counter_q_norm = 1 / counter_q_norm if counter_q_norm != 0 else 0
    #
    #     # normalization:
    #     for doc_id in doc_ranks.keys():
    #         doc_ranks[doc_id] = doc_ranks[doc_id] * counter_q_norm * 1 / (self.norm_anchor[doc_id])
    #
    #     sorted_docs = sorted([(doc_id, rank) for doc_id, rank in doc_ranks.items()], key=lambda x: x[1], reverse=True)[
    #                   :20]
    #     for doc_id, sim in sorted_docs:
    #         res.append((doc_id, sim))
    #     return res
    #
    # #     # END SOLUTION
    # #     # return jsonify(res)

    def backend_search(self, q):
        res = []

        english_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]
        RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        all_stopwords = english_stopwords.union(corpus_stopwords)
        query = [token.group().lower() for token in RE_WORD.finditer(q.lower()) if
                 token.group().lower() not in all_stopwords]

        query = " ".join(query)

        if len(query) == 0:
            return

        def title_search_task():
            return self.title_search(query)

        def body_search_task():
            return self.search_body(query)

        #
        # def anchor_search_task():
        #     return self.search_anchor(query)

        # Execute search functions concurrently with ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            title_results = executor.submit(title_search_task).result()
            body_results = executor.submit(body_search_task).result()
            # anchor_results = executor.submit(anchor_search_task).result()

        # Process results and update scores
        title_weight = 0.85
        body_weight = 0.15
        # anchor_weight = 0.05
        scores = Counter()
        scores.update({doc_id: (score * title_weight) for doc_id, score in title_results})
        scores.update({doc_id: (score * body_weight) for doc_id, score in body_results})
        scores.update({doc_id: score + math.log(self.page_rank[doc_id], 10)
                       for doc_id, score in scores.items() if doc_id in self.page_rank})
        # scores.update({doc_id: score * anchor_weight for doc_id, score in anchor_results})

        # Sort scores and retrieve top 100 results
        res = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
        res = [(str(doc_id), self.doc_title_pairs[doc_id]) for doc_id, score in res]
        return res
