from .corpus_algorithm import CORPUS_ALGORITHM
from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
import os
import csv
logger = ir_datasets.log.easy()

class VIEW():
    def __init__(self,
                 scorer: pt.Transformer,
                 corpus_graph: 'CorpusGraph',
                 view_location: str,
                 space_name: str,
                 view_name: str):
        self.scored_count = 0
        self.scorer = scorer
        self.corpus_graph = corpus_graph
        self.doc_location = {}
        self.view_name = view_name
        self.view_location = view_location
        self.space_name = space_name

    def create_view(self, df: pd.DataFrame, initial_size) -> pd.DataFrame:
        result = {'qid': [], 'query': [], 'docno': [],
                  'rank': [], 'score': []}

        locations = []
        doc_location_by_qid = {}

        df = dict(iter(df.groupby(by=['qid'])))
        qids = df.keys()
        qids = logger.pbar(qids, desc='adaptive re-ranking', unit='query')

        for qid in qids:
            query_df = df[qid].head(initial_size)

            # format: (row, column)
            self.doc_location = {}
            count = 0
            for index, doc in query_df.iterrows():
                self.doc_location[doc.docno] = [(count, 0)]
                count += 1

            query = query_df['query'].iloc[0]
            scores = {}
            self.score_algorithm(query_df, scores, qid, query)

            # Add scored items to results
            result['qid'].append(np.full(min(len(scores), 1000), qid))
            result['query'].append(np.full(min(len(scores), 1000), query))
            result['rank'].append(np.arange(min(len(scores), 1000)))
            for did, score in Counter(scores).most_common():
                cutoff_count = 0
                if cutoff_count < 1000:
                    result['docno'].append(did)
                    result['score'].append(score)
                    cutoff_count += 1
            doc_location_by_qid[qid[0]] = self.doc_location.copy()
        final = pd.DataFrame({
            'qid': np.concatenate(result['qid']),
            'query': np.concatenate(result['query']),
            'docno': result['docno'],
            'rank': np.concatenate(result['rank']),
            'score': result['score']
        })
        final.to_csv(f'{self.view_location}/{self.space_name}/{self.view_name}/results.csv')

        qids = np.concatenate(result['qid'])
        docnos = result['docno']
        for index, qid in enumerate(qids):
            locations.append(
                {'in_location': doc_location_by_qid[qid][docnos[index]], 'out_location': np.concatenate(result['rank'])[index]})
        newpath = f'{self.view_location}/{self.space_name}/{self.view_name}'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        with open(f'{self.view_location}/{self.space_name}/{self.view_name}/data.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                                    'in_location', 'out_location'])
            writer.writeheader()
            writer.writerows(locations)
        print('Total Documents Scored: ' + str(self.scored_count))

    def score_algorithm(self, batch, scores, qid, query):
        # default algorithm (score initial documents)
        self.scored_count += len(batch)
        scored = self.scorer(batch)

        scores.update({k: s for k, s in zip(scored.docno, scored.score)})


# View: initial documents and all of their neighbours
class VIEW1(VIEW):
    def __init__(self,
                 scorer: pt.Transformer,
                 corpus_graph: 'CorpusGraph',
                 view_location: str,
                 space_name: str,
                 view_name: str):
        super().__init__(scorer, corpus_graph, view_location, space_name, view_name)

    def score_algorithm(self, batch, scores, qid, query):
        # Score initial documents and all neighbours of initial documents
        to_score = {}
        to_score.update({k: 0 for k in batch.docno})
        for did in batch.docno:
            count = 1
            for target_did in self.corpus_graph.neighbours(did):
                to_score[target_did] = 0
                if target_did in self.doc_location:
                    self.doc_location[target_did].append(
                        (self.doc_location[did][0][0], count))
                else:
                    self.doc_location[target_did] = [
                        (self.doc_location[did][0][0], count)]
                count += 1
        to_score = pd.DataFrame(to_score.keys(), columns=['docno'])
        to_score['qid'] = [qid for i in range(len(to_score))]
        to_score['query'] = [query for i in range(len(to_score))]

        self.scored_count += len(to_score)
        scored = self.scorer(to_score)
        
        scores.update({k: s for k, s in zip(scored.docno, scored.score)})
