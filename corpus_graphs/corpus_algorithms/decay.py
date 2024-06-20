from .corpus_algorithm import CORPUS_ALGORITHM
from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
import csv
import math
logger = ir_datasets.log.easy()


class DECAY(CORPUS_ALGORITHM):
    def __init__(self,
                 scorer: pt.Transformer,
                 corpus_graph: 'CorpusGraph',
                 budget: int = 100,
                 verbose: bool = False,
                 metadata: str = ''):
        super().__init__(scorer, corpus_graph, budget=budget,
                         verbose=verbose, metadata=metadata)
        self.algorithm_type = 'decay'

    def score_algorithm(self, batch, scores, qid, query):
        # Score initial documents
        # Then score lexicographically while applying a decay to edge list size (reduces as algorithm traverses initial retrieved list)
        to_score = {}
        to_score.update(
            {k: 0 for k in batch.docno[:min(self.budget, len(batch.docno))]})
        remaining = self.budget - len(to_score.keys())

        k = len(self.corpus_graph.neighbours(batch.docno.iloc[0]))
        c = len(batch.docno)

        batch = batch.sort_values(by=['rank'])
        for index, did in enumerate(batch.docno):
            if remaining <= 0:
                break
            # Max number to score for the row: [1, k]
            num = self.decay(index, self.budget - c, k, c)
            count = 0
            for target_did in self.corpus_graph.neighbours(did):
                if remaining <= 0:
                    break
                if count <= num and target_did not in to_score:
                    to_score[target_did] = 0
                    remaining -= 1
                    count += 1

        to_score = pd.DataFrame(to_score.keys(), columns=['docno'])
        to_score['qid'] = [qid for i in range(len(to_score))]
        to_score['query'] = [query for i in range(len(to_score))]

        self.scored_count += len(to_score)
        scored = self.scorer(to_score)
        
        scores.update({k: s for k, s in zip(scored.docno, scored.score)})
    
    def decay(self, i, budget, k, c):
        return k


class LINEAR_DECAY(DECAY):
    def __init__(self,
                 scorer: pt.Transformer,
                 corpus_graph: 'CorpusGraph',
                 budget: int = 100,
                 verbose: bool = False,
                 metadata: str = ''):
        super().__init__(scorer, corpus_graph, budget=budget,
                         verbose=verbose, metadata=metadata)
        self.algorithm_type = 'linear_decay'

    def decay(self, i, budget, k, c):
        if budget >= (k * c):
            m = 0
        else:
            m = -0.5 * ((k * c) / budget)
        return max(math.ceil(k * (m * (i / c) + 1)), 1)

