import statistics
import math
from .corpus_algorithm import CORPUS_ALGORITHM
from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
import csv
logger = ir_datasets.log.easy()


class ADAPTIVE_THRESHOLD_CUTOFF_BACKFILL(CORPUS_ALGORITHM):
    def __init__(self,
                 scorer: pt.Transformer,
                 corpus_graph: 'CorpusGraph',
                 budget: int = 1000,
                 cutoff: int = 900,
                 k: int = 1,
                 verbose: bool = False,
                 metadata: str = ''):
        super().__init__(scorer, corpus_graph, budget=budget,
                         verbose=verbose, metadata=metadata)
        self.algorithm_type = f'adaptive_threshold_cutoff_backfill_{k}'
        self.k = k
        self.cutoff = cutoff

    def score_algorithm(self, batch, scores, qid, query):
        # Score top d initial retrieved documents (establishes threshold)
        # Alternate: score initial retrieved document, perform ladr_adaptive on neighbours of initial retrieved document until exhaustion (expanding only if document passes threshold)
        # Then scores 100 from initial retrieved (continues algorithm if exhausted)
        # Returns 900 to allow 100 backfilled retriever documents
        d = 50
        scored_list = []
        score_queue = {}
        batch_queue = {}
        scored_docs = {}
        to_score = {}

        to_score.update({k: 0 for k in batch.docno[:min(d, self.budget)]})
        to_score = pd.DataFrame(to_score.keys(), columns=['docno'])
        to_score['qid'] = [qid for i in range(len(to_score))]
        to_score['query'] = [query for i in range(len(to_score))]
        batch_scored = self.scorer(to_score)
        threshold = self.calculate_threshold(batch_scored, d)
        scored_list.append(batch_scored)
        score_queue.update(dict(zip(batch_scored.docno, batch_scored.score)))
        scored_docs.update(dict.fromkeys(batch_scored.docno, 0))

        batch_queue.update(dict(zip(batch.docno, batch.score)))

        remaining = self.budget - len(to_score.docno)

        while remaining > 0:
            to_score = {}
            if remaining <= 100 and len(batch_queue) > 0:
                did = max(batch_queue, key=batch_queue.get)
                del batch_queue[did]
                if did not in scored_docs:
                    to_score = pd.DataFrame([did], columns=['docno'])
                    to_score['qid'] = [qid]
                    to_score['query'] = [query]
                    batch_scored = self.scorer(to_score)
                    remaining -= 1
                    scored_list.append(batch_scored)
                    if batch_scored.iloc[0].score >= threshold:
                        score_queue.update(
                            dict(zip(batch_scored.docno, batch_scored.score)))
                    scored_docs.update(dict.fromkeys(batch_scored.docno, 0))
            elif len(score_queue) > 0:
                did = max(score_queue, key=score_queue.get)
                del score_queue[did]
                k_count = 0
                for target_did in self.corpus_graph.neighbours(did):
                    if remaining <= 0 or k_count >= self.k:
                        break
                    if target_did not in scored_docs and target_did not in to_score:
                        to_score[target_did] = 0
                        remaining -= 1
                        k_count += 1
                if len(to_score.keys()) > 0:
                    to_score = pd.DataFrame(to_score.keys(), columns=['docno'])
                    to_score['qid'] = [qid for i in range(len(to_score))]
                    to_score['query'] = [query for i in range(len(to_score))]
                    batch_scored = self.scorer(to_score)
                    scored_list.append(batch_scored)
                    score_queue.update(dict(
                        [x for x in zip(batch_scored.docno, batch_scored.score) if x[1] >= threshold]))
                    scored_docs.update(dict.fromkeys(batch_scored.docno, 0))
            else:
                if len(batch_queue) == 0:
                    break
                did = max(batch_queue, key=batch_queue.get)
                del batch_queue[did]
                if did not in scored_docs:
                    to_score = pd.DataFrame([did], columns=['docno'])
                    to_score['qid'] = [qid]
                    to_score['query'] = [query]
                    batch_scored = self.scorer(to_score)
                    remaining -= 1
                    scored_list.append(batch_scored)
                    if batch_scored.iloc[0].score >= threshold:
                        score_queue.update(
                            dict(zip(batch_scored.docno, batch_scored.score)))
                    scored_docs.update(dict.fromkeys(batch_scored.docno, 0))

        scored = pd.concat(scored_list)
        self.scored_count += len(scored)
        scored = scored.sort_values(
            by=['score'], ascending=False).iloc[:self.cutoff]
        scores.update({k: s for k, s in zip(scored.docno, scored.score)})

    def calculate_threshold(self, batch, d):
        return 0


class ADAPTIVE_THRESHOLD_CUTOFF_BACKFILL_MEAN(ADAPTIVE_THRESHOLD_CUTOFF_BACKFILL):
    def __init__(self,
                 scorer: pt.Transformer,
                 corpus_graph: 'CorpusGraph',
                 budget: int = 1000,
                 cutoff: int = 900,
                 k: int = 1,
                 verbose: bool = False,
                 metadata: str = ''):
        super().__init__(scorer, corpus_graph, budget=budget, cutoff=cutoff, k=k,
                         verbose=verbose, metadata=metadata)
        self.algorithm_type = f'adaptive_threshold_cutoff_backfill_mean_{k}'

    def calculate_threshold(self, batch, d):
        return statistics.mean(batch.score[:d])
