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
                 batch_size: Optional[int] = None,
                 verbose: bool = False,
                 metadata: str = ''):
        super().__init__(scorer, corpus_graph, budget=budget,
                         batch_size=batch_size, verbose=verbose, metadata=metadata)
        self.algorithm_type = 'ladr_adaptive'

    def score_algorithm(self, batch, scores, qid, query):
        # Score initial documents
        # Repeatedly score k neighbours of top c scored documents
        scored_list = []
        score_queue = {}
        scored_docs = {}
        to_score = {}

        to_score.update(
            {k: 0 for k in batch.docno[:min(self.budget, len(batch.docno))]})
        to_score = pd.DataFrame(to_score.keys(), columns=['docno'])
        to_score['qid'] = [qid for i in range(len(to_score))]
        to_score['query'] = [query for i in range(len(to_score))]
        batch_scored = self.scorer(to_score)
        scored_list.append(batch_scored)
        score_queue.update(dict(zip(batch_scored.docno, batch_scored.score)))
        scored_docs.update(dict.fromkeys(batch_scored.docno, 0))

        remaining = self.budget - len(to_score.keys())

        while remaining > 0:
            to_score = {}
            did = max(score_queue, key=score_queue.get)
            del score_queue[did]
            for target_did in self.corpus_graph.neighbours(did):
                if remaining <= 0:
                    break
                if target_did not in scored_docs:
                    to_score[target_did] = 0
                    remaining -= 1
            if len(to_score.keys()) == 0:
                break
            to_score = pd.DataFrame(to_score.keys(), columns=['docno'])
            to_score['qid'] = [qid for i in range(len(to_score))]
            to_score['query'] = [query for i in range(len(to_score))]
            batch_scored = self.scorer(to_score)
            scored_list.append(batch_scored)
            score_queue.update(dict(zip(batch_scored.docno, batch_scored.score)))
            scored_docs.update(dict.fromkeys(batch_scored.docno, 0))

        
        scored = pd.concat(scored_list)
        self.scored_count += len(scored)
        scores.update({k: s for k, s in zip(scored.docno, scored.score)})
