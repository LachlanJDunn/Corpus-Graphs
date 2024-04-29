from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
import csv
logger = ir_datasets.log.easy()


class GAR(pt.Transformer):
    """
    A transformer that implements the Graph-based Adaptive Re-ranker algorithm from
    MacAvaney et al. "Adaptive Re-Ranking with a Corpus Graph" CIKM 2022.

    Required input columns: ['qid', 'query', 'docno', 'score', 'rank']
    Output columns: ['qid', 'query', 'docno', 'score', 'rank', 'iteration']
    where iteration defines the batch number which identified the document. Specifically
    even=initial retrieval   odd=corpus graph    -1=backfilled
    
    """

    def __init__(self,
                 scorer: pt.Transformer,
                 corpus_graph: 'CorpusGraph',
                 num_results: int = 1000,
                 batch_size: Optional[int] = None,
                 backfill: bool = True,
                 enabled: bool = True,
                 verbose: bool = False,
                 collect_data: bool = False):
        """
            GAR init method
            Args:
                scorer(pyterrier.Transformer): A transformer that scores query-document pairs. It will only be provided with ['qid, 'query', 'docno', 'score'].
                corpus_graph(pyterrier_adaptive.CorpusGraph): A graph of the corpus, enabling quick lookups of nearest neighbours
                num_results(int): The maximum number of documents to score (called "budget" and $c$ in the paper)
                batch_size(int): The number of documents to score at once (called $b$ in the paper). If not provided, will attempt to use the batch size from the scorer
                backfill(bool): If True, always include all documents from the initial stage, even if they were not re-scored
                enabled(bool): If False, perform re-ranking without using the corpus graph
                verbose(bool): If True, print progress information
        """
        self.collect_data = collect_data
        self.scorer = scorer
        self.corpus_graph = corpus_graph
        self.num_results = num_results
        if batch_size is None:
            batch_size = scorer.batch_size if hasattr(
                scorer, 'batch_size') else 16
        self.batch_size = batch_size
        self.backfill = backfill
        self.enabled = enabled
        self.verbose = verbose

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Graph-based Adaptive Re-ranking to the provided dataframe. Essentially,
        Algorithm 1 from the paper.
        """
        result = {'qid': [], 'query': [], 'docno': [],
                  'rank': [], 'score': [], 'iteration': []}

        locations = []
        doc_location_by_qid = {}

        # provides dictionary of qid: data (ie. query, rank, score etc.)
        df = dict(iter(df.groupby(by=['qid'])))
        qids = df.keys()

        # adds progress bar
        if self.verbose:
            qids = logger.pbar(qids, desc='adaptive re-ranking', unit='query')

        for qid in qids:
            if self.collect_data:
                # format: (row, column)
                doc_location = {}
                count = 0
                for index, doc in df[qid].iterrows():
                    doc_location[doc.docno] = (count, 0)
                    count += 1
            query = df[qid]['query'].iloc[0]
            scores = {}
            # initial results (from first round)
            res_map = [Counter(dict(zip(df[qid].docno, df[qid].score)))]
            if self.enabled:
                res_map.append(Counter()) 
            
            batch = self.scorer(df[qid])
            scores.update({k: (s, 0)
                           for k, s in zip(batch.docno, batch.score)})
            self._drop_docnos_from_counters(batch.docno, res_map)
            self._update_frontier(df[qid], res_map[1], doc_location)
            batch = pd.DataFrame(res_map[1].most_common(), columns=['docno', 'score'])
            batch['qid'] = [qid for i in range(len(batch.index))]
            batch['query'] = [query for i in range(len(batch.index))]
            batch = self.scorer(batch)
            scores.update({k: (s, 0)
                           for k, s in zip(batch.docno, batch.score)})
            self._drop_docnos_from_counters(batch.docno, res_map)

            # Add scored items to results
            result['qid'].append(np.full(len(scores), qid))
            result['query'].append(np.full(len(scores), query))
            result['rank'].append(np.arange(len(scores)))
            for did, (score, i) in Counter(scores).most_common():
                result['docno'].append(did)
                result['score'].append(score)
                result['iteration'].append(i)

            # Backfill unscored items
            if self.backfill and len(scores) < self.num_results:
                last_score = result['score'][-1] if result['score'] else 0.
                count = min(self.num_results - len(scores), len(res_map[0]))
                result['qid'].append(np.full(count, qid))
                result['query'].append(np.full(count, query))
                result['rank'].append(
                    np.arange(len(scores), len(scores) + count))
                for i, (did, score) in enumerate(res_map[0].most_common()):
                    if i >= count:
                        break
                    result['docno'].append(did)
                    result['score'].append(last_score - 1 - i)
                    result['iteration'].append(-1)
            if self.collect_data:
                doc_location_by_qid[qid[0]] = doc_location
        if self.collect_data:
            qids = np.concatenate(result['qid'])
            docnos = result['docno']
            for index, qid in enumerate(qids):
                locations.append(
                    {'in_location': doc_location_by_qid[qid][docnos[index]], 'out_location': np.concatenate(result['rank'])[index]})
            with open('location_data.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                                        'in_location', 'out_location'])
                writer.writeheader()
                writer.writerows(locations)
        return pd.DataFrame({
            'qid': np.concatenate(result['qid']),
            'query': np.concatenate(result['query']),
            'docno': result['docno'],
            'rank': np.concatenate(result['rank']),
            'score': result['score'],
            'iteration': result['iteration'],
        })

    def _update_frontier(self, batch, frontier, doc_location):
        for did in batch.docno:
            count = 1
            for target_did in self.corpus_graph.neighbours(did):
                frontier[target_did] = 0
                if self.collect_data and target_did not in doc_location:
                    doc_location[target_did] = (doc_location[did][0], count)
                count += 1

    def _drop_docnos_from_counters(self, docnos, counters):
        # remove scored documents from frontier/inital list
        for docno in docnos:
            for c in counters:
                del c[docno]
