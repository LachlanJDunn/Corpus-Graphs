from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
import csv
import os
logger = ir_datasets.log.easy()


class CORPUS_ALGORITHM(pt.Transformer):
    def __init__(self,
                 scorer: pt.Transformer,
                 corpus_graph: 'CorpusGraph',
                 budget: int = 1000,
                 verbose: bool = False,
                 metadata: str = ''):
        self.scored_count = 0
        self.scorer = scorer
        self.corpus_graph = corpus_graph
        self.budget = budget
        self.metadata = metadata
        self.verbose = verbose
        self.algorithm_type = 'corpus_algorithm'
        self.doc_location = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = {'qid': [], 'query': [], 'docno': [],
                  'rank': [], 'score': []}

        df = dict(iter(df.groupby(by=['qid'])))
        qids = df.keys()
        if self.verbose:
            qids = logger.pbar(qids, desc='adaptive re-ranking', unit='query')

        for qid in qids:
            query = df[qid]['query'].iloc[0]
            scores = {}
            self.score_algorithm(df[qid], scores, qid, query)

            # Add scored items to results
            result['qid'].append(np.full(len(scores), qid))
            result['query'].append(np.full(len(scores), query))
            result['rank'].append(np.arange(len(scores)))
            for did, score in Counter(scores).most_common():
                result['docno'].append(did)
                result['score'].append(score)
        if self.metadata != '':
            if not os.path.exists(self.metadata):
                os.makedirs(self.metadata)
            with open(f'{self.metadata}/{self.algorithm_type}_metadata.txt', 'a') as file:
                file.write(f'{self.budget} {self.scored_count}\n')
        print('Total Documents Scored: ' + str(self.scored_count))
        return pd.DataFrame({
            'qid': np.concatenate(result['qid']),
            'query': np.concatenate(result['query']),
            'docno': result['docno'],
            'rank': np.concatenate(result['rank']),
            'score': result['score']
        })

    def score_algorithm(self, batch, scores, qid, query):
        # default algorithm (score initial documents)
        self.scored_count += len(batch)
        scored = self.scorer(batch)

        scores.update({k: s for k, s in zip(scored.docno, scored.score)})
