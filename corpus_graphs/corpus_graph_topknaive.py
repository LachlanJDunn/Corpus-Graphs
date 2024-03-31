import pickle
import tempfile
from lz4.frame import LZ4FrameFile
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import more_itertools
from typing import Union, Tuple, List
import ir_datasets
from npids import Lookup
import IPython


try:
  from functools import cached_property
except ImportError:
  # polyfill for python<3.8, adapted from <https://github.com/pydanny/cached-property>
  class cached_property(object):
    def __init__(self, fn):
      self.fn = fn

    def __get__(self, obj, cls):
      value = obj.__dict__[self.fn.__name__] = self.fn(obj)
      return value

logger = ir_datasets.log.easy()


class CorpusGraph:
  def neighbours(self, docid: Union[int, str]) -> Union[np.array, List[str], Tuple[np.array, np.array], Tuple[List[str], np.array]]:
    raise NotImplementedError()

  #
  @staticmethod
  def load(path, **kwargs):
    # checks requirements and loads graph from path
    with (Path(path)/'pt_meta.json').open('rt') as fin:
      meta = json.load(fin)
    assert meta.get('type') == 'corpus_graph'
    fmt = meta.get('format')
    if fmt == 'np_topk' or fmt == 'numpy_kmax':
      return NpTopKCorpusGraph(path, **kwargs)
    raise ValueError(f'Unknown corpus graph format: {fmt}')

  @staticmethod
  def from_dataset(dataset: str, variant, version: str = 'latest', **kwargs):
    from pyterrier.batchretrieve import _from_dataset
    return _from_dataset(dataset, variant=variant, version=version, clz=CorpusGraph.load, **kwargs)

  @staticmethod
  def from_retriever(retriever, docs_it, out_dir: Union[str, Path], k: int = 16, batch_size: int = 1024) -> 'CorpusGraph':
    out_dir = Path(out_dir)
    if out_dir.exists():
      raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    edges_path = out_dir/'edges.u32.np'

    doc_size = 0
    id_list = []
    id_count = 0
    scored_count = 0

    # First step: We need a docno <-> index mapping for this to work. Do a pass over the iterator
    # to build a docno loookup file, while also writing the contents to a temporary file that we'll read
    # in the next loop. (This avoids loading the entire iterator into memory.)
    with tempfile.TemporaryDirectory() as dout:  # create temp directory to store
      # write to temp file and lookup outdir
      with LZ4FrameFile(f'{dout}/docs.pkl.lz4', 'wb') as fout:
        for doc in logger.pbar(docs_it, miniters=1, smoothing=0, desc='first pass'):
          pickle.dump(doc, fout)  # write serialized docs to temp file
          id_list.append(doc['docno'])
      doc_size = len(id_list)
      ids = pd.DataFrame(data={'id': [-1 for i in range(doc_size)]}, index=id_list)

      # Perform retrieval in chunks for efficiency
      with ir_datasets.util.finialized_file(str(edges_path), 'wb') as fe, LZ4FrameFile(f'{dout}/docs.pkl.lz4', 'rb') as fin, Lookup.builder(out_dir/'docnos.npids') as docnos:
        for chunk in more_itertools.chunked(logger.pbar(range(doc_size), miniters=1, smoothing=0, desc='searching', total=doc_size), batch_size):
          chunk = [pickle.load(fin) for _ in chunk]  # creates list of docs
          chunk_df = pd.DataFrame(chunk).rename(
              columns={'docno': 'qid', 'text': 'query'})
          to_drop = ids.loc[[i for i in chunk_df.qid.to_numpy()]]
          to_drop = to_drop.loc[to_drop['id'] != -1].index
          # remove already scored documents
          chunk_df = chunk_df.loc[~chunk_df['qid'].isin(to_drop)]
          scored_count += len(chunk_df.index)
          res = retriever(chunk_df)  # result of retrieval of one chunk
          # mapping of qid to query/docno/docno/score/rank (as multiple queries in chunk)
          res_by_qid = dict(iter(res.groupby('qid')))
          for docno in chunk_df['qid']:  # loops through docnos in chunk
            new_docs_count = 0
            # results for one query (ie. document)
            did_res = res_by_qid.get(docno)
            dids = []
            if did_res is not None:
              # top k results
              did_res = did_res.iloc[:k]
              append_doc = False
              if docno not in did_res['docno'].to_numpy() and ids.loc[docno, 'id'] == -1:
                append_doc = True
              if len(did_res) > 0:
                for i in range(len(did_res.index)):
                  if append_doc == True and i == len(did_res.index) - 1:
                    docnos.add(docno)
                    new_docs_count += 1
                    ids.loc[docno, 'id'] = id_count
                    dids.append(id_count)
                    id_count += 1
                  else:
                    docno2 = did_res.iloc[i]['docno']
                    if ids.loc[docno2, 'id'] == -1:
                        docnos.add(docno2)
                        new_docs_count += 1
                        ids.loc[docno2, 'id'] = id_count
                        dids.append(id_count)
                        id_count += 1
                    else:
                        dids.append(ids.loc[docno2, 'id'])
            # pad missing docids
            if len(dids) < k:
              if did_res is None:
                dids += [id_count] * (k - len(dids))
              else:
                dids += [id_count - len(did_res.index)] * (k - len(dids))
            # write neighbours
            for _ in range(new_docs_count):
                fe.write(np.array(dids, dtype=np.uint32).tobytes())
    print(scored_count)
    # Finally, keep track of metadata about this artefact.
    with (out_dir/'pt_meta.json').open('wt') as fout:
      json.dump({
          'type': 'corpus_graph',
          'format': 'np_topk',
          'doc_count': doc_size,
          'k': k,
      }, fout)

    return NpTopKCorpusGraph(out_dir)


class NpTopKCorpusGraph(CorpusGraph):
  def __init__(self, path, k=None):
    self.path = Path(path)
    with (self.path/'pt_meta.json').open('rt') as fin:
      self.meta = json.load(fin)
    assert self.meta.get('type') == 'corpus_graph' and self.meta.get(
        'format') in ('numpy_kmax', 'np_topk')
    self._data_k = self.meta['k']
    if k is not None:
      assert k <= self.meta['k']
    self._k = self.meta['k'] if k is None else k
    self._edges_path = self.path/'edges.u32.np'
    self._docnos = Lookup(self.path/'docnos.npids')

  def __repr__(self):
    return f'NpTopKCorpusGraph({repr(str(self.path))}, k={self._k})'

  @cached_property
  def edges_data(self):
    # loads data and alters to fit k
    res = np.memmap(self._edges_path, mode='r',
                    dtype=np.uint32).reshape(-1, self._data_k)
    if self._k != self._data_k:
      res = res[:, :self._k]
    return res

  def neighbours(self, docid):
    # returns neighbours using data
    as_str = isinstance(docid, str)
    if as_str:
      docid = self._docnos.inv[docid]
    neigh = self.edges_data[docid]
    if as_str:
      neigh = self._docnos.fwd[neigh]
    return neigh

  def to_limit_k(self, k: int) -> 'NpTopKCorpusGraph':
    """
    Creates a version of the graph with a maximum of k edges from each node.
    k must be less than the number of edges per node from the original graph.
    """
    return NpTopKCorpusGraph(self.path, k)
