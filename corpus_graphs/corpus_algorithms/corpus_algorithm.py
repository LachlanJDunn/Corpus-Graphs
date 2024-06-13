from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
import csv
logger = ir_datasets.log.easy()


class CORPUS_ALGORITHM(pt.Transformer):
    def __init__(self,
                 scorer: pt.Transformer,
                 corpus_graph: 'CorpusGraph',
                 budget: int = 1000,
                 batch_size: Optional[int] = None,
                 verbose: bool = False,
                 metadata: str = ''):
        """
            Corpus Algorithm init method
            Args:
                scorer(pyterrier.Transformer): A transformer that scores query-document pairs. It will only be provided with ['qid, 'query', 'docno', 'score'].
                corpus_graph(pyterrier_adaptive.CorpusGraph): A graph of the corpus, enabling quick lookups of nearest neighbours
                budget(int): The maximum number of extra documents to score (0: score initial documents only; <0: infinite budget)
                batch_size(int): The number of documents to score at once. If not provided, will attempt to use the batch size from the scorer
                verbose(bool): If True, print progress information
                metadata(str): Location to store metadata (if requested)
        """
        self.scored_count = 0
        self.scorer = scorer
        self.corpus_graph = corpus_graph
        self.budget = budget
        self.metadata = metadata
        if batch_size is None:
            batch_size = scorer.batch_size if hasattr(
                scorer, 'batch_size') else 16
        self.batch_size = batch_size
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
            with open(self.metadata, 'a') as file:
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
        scored = self.scorer(batch)
        self.scored_count += len(scored)
        scores.update({k: s for k, s in zip(scored.docno, scored.score)})
