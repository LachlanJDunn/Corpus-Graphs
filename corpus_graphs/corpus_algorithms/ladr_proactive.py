from .corpus_algorithm import CORPUS_ALGORITHM
from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
import csv
logger = ir_datasets.log.easy()


class LADR_PROACTIVE(CORPUS_ALGORITHM):
    def __init__(self,
                 scorer: pt.Transformer,
                 corpus_graph: 'CorpusGraph',
                 budget: int = 1000,
                 batch_size: Optional[int] = None,
                 verbose: bool = False,
                 collect_data: bool = False):
        super().__init__(scorer, corpus_graph, budget=budget,
                       batch_size=batch_size, verbose=verbose, collect_data=collect_data)
        self.algorithm_type = 'ladr_proactive'

    def score_algorithm(self, batch, scores, qid, query):
        # Score initial documents and all neighbours of initial documents
        to_score = {}
        to_score.update({k: 0 for k in batch.docno})
        for did in batch.docno:
            count = 1
            for target_did in self.corpus_graph.neighbours(did):
                to_score[target_did] = 0
                if self.collect_data:
                    if target_did in self.doc_location:
                        self.doc_location[target_did].append((self.doc_location[did][0][0], count))
                    else:
                        self.doc_location[target_did] = [(self.doc_location[did][0][0], count)]
                count += 1
        to_score = pd.DataFrame(to_score.keys(), columns=['docno'])
        to_score['qid'] = [qid for i in range(len(to_score))]
        to_score['query'] = [query for i in range(len(to_score))]
        
        scored = self.scorer(to_score)
        self.scored_count += len(scored)
        scores.update({k: s for k, s in zip(scored.docno, scored.score)})