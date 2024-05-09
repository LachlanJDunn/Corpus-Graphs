import CORPUS_ALGORITHM
from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
import csv
logger = ir_datasets.log.easy()


class LADR_PROACTIVE(CORPUS_ALGORITHM):
    def __init__(self):
        super.__init__()

    def score_algorithm(self, batch, scores, qid, query):
        # Score initial documents and all neighbours of initial documents
        to_score = {}
        to_score.update({k: 0 for k in batch.docno})
        for did in batch.docno:
            count = 1
            for target_did in self.corpus_graph.neighbours(did):
                to_score[target_did] = 0
                if self.collect_data and target_did not in self.doc_location:
                    self.doc_location[target_did] = (
                        self.doc_location[did][0], count)
                count += 1
        to_score = pd.DataFrame(to_score.keys(), columns=['docno'])
        to_score['qid'] = [qid for i in range(len(batch.index))]
        to_score['query'] = [query for i in range(len(batch.index))]
        
        scored = self.scorer(to_score)
        self.scored_count += len(scored)
        scores.update({k: s for k, s in zip(scored.docno, scored.score)})