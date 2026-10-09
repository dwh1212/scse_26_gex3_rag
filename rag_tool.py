## THIS FILE COMPLETES THE RETRIEVAL TOOL.
## It loads the policy documents with policy_loader.py and exposes a
## dependency-free BM25 retriever that returns the most relevant excerpts.

import math
import re
from collections import Counter

from policy_loader import load_policy_documents

## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """Small, self-contained BM25 retriever over the loaded policy documents."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.corpus = [_tokenize(d.page_content) for d in docs]
        self.avgdl = sum(len(t) for t in self.corpus) / max(len(self.corpus), 1)
        self.idf = self._idf()

    def _idf(self):
        n = len(self.corpus)
        df = Counter()
        for tokens in self.corpus:
            df.update(set(tokens))
        return {term: math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5)) for term in df}

    def _score(self, query_tokens):
        scores = []
        for tokens in self.corpus:
            dl = len(tokens)
            tf = Counter(tokens)
            total = 0.0
            for q in set(query_tokens):
                if q not in self.idf:
                    continue
                f = tf.get(q, 0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                total += self.idf[q] * (f * (self.k1 + 1)) / denom
            scores.append(total)
        return scores

    def retrieve(self, query, top_k=3):
        query_tokens = _tokenize(query)
        scores = self._score(query_tokens)
        order = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.docs[i] for i in order]


def retrieve(query, top_k=3):
    """Return the top_k most relevant policy documents for the query."""
    retriever = BM25Retriever(documents)
    return retriever.retrieve(query, top_k=top_k)
