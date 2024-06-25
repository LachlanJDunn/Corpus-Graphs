from .corpus_algorithm import CORPUS_ALGORITHM
from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
import csv
logger = ir_datasets.log.easy()


class LADR_ADAPTIVE(CORPUS_ALGORITHM):
    def __init__(self,
                 scorer: pt.Transformer,
                 corpus_graph: 'CorpusGraph',
                 budget: int = 1000,
                 k: int = 1,
                 c: int = 1,
                 verbose: bool = False,
                 metadata: str = ''):
        super().__init__(scorer, corpus_graph, budget=budget,
                         verbose=verbose, metadata=metadata)
        self.algorithm_type = f'ladr_adaptive_{k}_{c}'
        self.k = k
        self.c = c

    def score_algorithm(self, batch, scores, qid, query):
        # Score initial documents
        # Score k neighbours of top c documents until no new documents added
        scored_docs = pd.DataFrame(columns=['docno', 'score'])
        to_score = {}

        to_score.update(
            {k: 0 for k in batch.docno[:min(self.budget, len(batch.docno))]})
        to_score = pd.DataFrame(to_score.keys(), columns=['docno'])
        to_score['qid'] = [qid for i in range(len(to_score))]
        to_score['query'] = [query for i in range(len(to_score))]
        batch_scored = self.scorer(to_score)
        scored_docs = pd.concat([scored_docs, batch_scored[['docno', 'score']]])

        remaining = self.budget - len(to_score)

        while remaining > 0:
            to_score = {}
            for did in scored_docs.sort_values(by=['score'], ascending=False).docno.iloc[:self.c]:
                if remaining <= 0:
                    break
                k_count = 0
                for target_did in self.corpus_graph.neighbours(did):
                    k_count += 1
                    if k_count > self.k or remaining <= 0:
                        break
                    if target_did not in scored_docs.docno and target_did not in to_score:
                        to_score[target_did] = 0
                        remaining -= 1
            if len(to_score.keys()) == 0:
                break
            to_score = pd.DataFrame(to_score.keys(), columns=['docno'])
            to_score['qid'] = [qid for i in range(len(to_score))]
            to_score['query'] = [query for i in range(len(to_score))]
            batch_scored = self.scorer(to_score)
            scored_docs = pd.concat([scored_docs, batch_scored[['docno', 'score']]])

        
        self.scored_count += len(scored_docs)
        scores.update({k: s for k, s in zip(scored_docs.docno, scored_docs.score)})
