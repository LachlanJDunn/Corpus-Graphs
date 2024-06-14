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
                 batch_size: Optional[int] = None,
                 verbose: bool = False,
                 metadata: str = ''):
        super().__init__(scorer, corpus_graph, budget=budget,
                         batch_size=batch_size, verbose=verbose, metadata=metadata)
        self.algorithm_type = 'decay'

    def score_algorithm(self, batch, scores, qid, query):
        # Score lexicographically while applying a decay to edge list size (reduces as algorithm traverses initial retrieved list)
        to_score = {}
        if len(batch.docno == 0):
            return
        k = len(self.corpus_graph.neighbours(batch.docno[0]))
        c = len(batch.docno)

        batch = batch.sort_values(by=['rank'])
        remaining = self.budget
        decay_factor = min(self.budget / (k * c), 1)
        count_list = []
        for index, did in enumerate(batch.docno):
            if remaining <= 0:
                break
            to_score[did] = 0
            remaining -= 1
            num = math.ceil(k * (decay_factor ^ index))
            for target_did in self.corpus_graph.neighbours(did):
                count = 1
                if remaining <= 0:
                    break
                if count < num and target_did not in to_score:
                    to_score[target_did] = 0
                    remaining -= 1
                    count += 1
                count_list.append(count)

        to_score = pd.DataFrame(to_score.keys(), columns=['docno'])
        to_score['qid'] = [qid for i in range(len(to_score))]
        to_score['query'] = [query for i in range(len(to_score))]

        scored = self.scorer(to_score)
        self.scored_count += len(scored)
        scores.update({k: s for k, s in zip(scored.docno, scored.score)})
        print(count_list)

