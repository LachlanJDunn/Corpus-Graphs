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
        budget: int = 1000,
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
        self.num_results = budget
        if batch_size is None:
            batch_size = scorer.batch_size if hasattr(scorer, 'batch_size') else 16
        self.batch_size = batch_size
        self.backfill = backfill
        self.enabled = enabled
        self.verbose = verbose

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Graph-based Adaptive Re-ranking to the provided dataframe. Essentially,
        Algorithm 1 from the paper.
        """
        result = {'qid': [], 'query': [], 'docno': [], 'rank': [], 'score': [], 'iteration': []}

        locations = []
        doc_location_by_qid = {}


        #provides dictionary of qid: data (ie. query, rank, score etc.)
        df = dict(iter(df.groupby(by=['qid'])))
        qids = df.keys()

        #adds progress bar
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
            res_map = [Counter(dict(zip(df[qid].docno, df[qid].score)))] # initial results (from first round)
            if self.enabled:
                res_map.append(Counter()) # frontier
            frontier_data = {'minscore': float('inf')} #ignores lower scored documents if budget is exceeded
            iteration = 0
            #while not all rescored and either frontier/initial retrieval not empty
            while len(scores) < self.num_results and any(r for r in res_map):
                if len(res_map[iteration%len(res_map)]) == 0:
                    # skip if results map / frontier is empty (and go to next one)
                    iteration += 1
                    continue
                this_res = res_map[iteration%len(res_map)] # alternate between the initial ranking and frontier (dict from id to score)
                size = min(self.batch_size, self.num_results - len(scores)) # get either the batch size or remaining budget (whichever is smaller)
                # build batch of documents to score in this round
                batch = this_res.most_common(size) #takes size number of documents and orders by highest score 
                batch = pd.DataFrame(batch, columns=['docno', 'score'])
                batch['qid'] = [qid for i in range(len(batch.index))]  # labels with qid and query
                batch['query'] = [query for i in range(len(batch.index))]

                # go score the batch of document with the re-ranker
                batch = self.scorer(batch)
                scores.update({k: (s, iteration) for k, s in zip(batch.docno, batch.score)}) #update information in form: docno: (score, iteration)
                self._drop_docnos_from_counters(batch.docno, res_map)
                if len(scores) < self.num_results and self.enabled: #if more documents to rescore
                    self._update_frontier(batch, res_map[1], frontier_data, scores, doc_location)
                iteration += 1

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
                result['rank'].append(np.arange(len(scores), len(scores) + count))
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
                writer = csv.DictWriter(csvfile, fieldnames=['in_location', 'out_location'])
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

    def _update_frontier(self, scored_batch, frontier, frontier_data, scored_dids, doc_location):
        remaining_budget = self.num_results - len(scored_dids) #total number of documents remaining to score
        for score, did in sorted(zip(scored_batch.score, scored_batch.docno), reverse=True): #loop through highest to lowest scores
            if len(frontier) < remaining_budget or score >= frontier_data['minscore']: #if more space or score higher than previous lowest added score
                hit = False
                count = 1
                for target_did in self.corpus_graph.neighbours(did): #loop through neighbours
                    if target_did not in scored_dids:
                        # updates score of neighbour to highest score of its already scored neighbours
                        if target_did not in frontier or score > frontier[target_did]:
                            frontier[target_did] = score
                            hit = True
                            if self.collect_data and target_did not in doc_location:
                                doc_location[target_did] = (doc_location[did][0], count)
                    count += 1
                if hit and score < frontier_data['minscore']:
                    frontier_data['minscore'] = score

    def _drop_docnos_from_counters(self, docnos, counters):
        # remove scored documents from frontier/inital list
        for docno in docnos:
            for c in counters:
                del c[docno]
