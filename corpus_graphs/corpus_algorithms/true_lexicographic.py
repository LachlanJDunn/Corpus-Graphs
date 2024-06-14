from .corpus_algorithm import CORPUS_ALGORITHM
from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
import csv
logger = ir_datasets.log.easy()


class TRUE_LEXICOGRAPHIC(CORPUS_ALGORITHM):
    def __init__(self,
                 scorer: pt.Transformer,
                 corpus_graph: 'CorpusGraph',
                 budget: int = 100,
                 batch_size: Optional[int] = None,
                 verbose: bool = False,
                 metadata: str = ''):
        super().__init__(scorer, corpus_graph, budget=budget,
                         batch_size=batch_size, verbose=verbose, metadata=metadata)
        self.algorithm_type = 'true_lexicographic'

    def score_algorithm(self, batch, scores, qid, query):
        # Score documents lexicographically based on ordering (position in initial list, position in edgelist)
        to_score = {}
        
        batch = batch.sort_values(by=['rank'])
        remaining = self.budget
        for did in batch.docno:
            if remaining <= 0:
                break
            to_score[did] = 0
            remaining -= 1
            for target_did in self.corpus_graph.neighbours(did):
                if remaining <= 0:
                    break
                if target_did not in to_score:
                    to_score[target_did] = 0
                    remaining -= 1

        to_score = pd.DataFrame(to_score.keys(), columns=['docno'])
        to_score['qid'] = [qid for i in range(len(to_score))]
        to_score['query'] = [query for i in range(len(to_score))]

        self.scored_count += len(to_score.keys())
        scored = self.scorer(to_score)

        scores.update({k: s for k, s in zip(scored.docno, scored.score)})
