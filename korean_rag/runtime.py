"""Portable, manifest-verified sparse/dense retrieval and extractive QA runtime.

Reader windows follow retrieved source offsets, never annotated answer offsets.
All model and tokenizer loading is local; the serving path never loads pickle.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import time
import re
import os
from pathlib import PurePosixPath
from collections.abc import Sequence
import numpy as np

from .retrieval import mean_pooling, reciprocal_rank_fusion, tokenize_for_bm25, top_k_from_scores
from .reranking import CrossEncoderReranker
from .validation import validate_question_tokens


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_bundle(directory: Path) -> dict:
    directory = Path(directory)
    if directory.is_symlink() or (hasattr(directory, 'is_junction') and directory.is_junction()):
        raise ValueError('The artifact root must not be a filesystem link.')
    directory = directory.resolve()
    if (directory/'manifest.json').is_symlink():
        raise ValueError('The artifact manifest must not be a filesystem link.')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('files'), dict):
        raise ValueError('Unsupported or incomplete artifact manifest.')
    for name, record in manifest['files'].items():
        if not isinstance(name, str) or not name or any(c in name for c in '\\:<>"|?*') or PurePosixPath(name).is_absolute() or any(p in {'','.','..'} for p in name.split('/')):
            raise ValueError('Artifact paths must be unambiguous relative paths.')
        if not isinstance(record, dict) or not isinstance(record.get('bytes'), int) or isinstance(record['bytes'], bool) or record['bytes'] < 0 or not re.fullmatch(r'[0-9a-f]{64}', str(record.get('sha256',''))):
            raise ValueError('Artifact sizes and hashes must be valid.')
        relative = Path(name)
        unresolved = directory / relative
        for item in (unresolved, *unresolved.parents):
            if item == directory.parent:
                break
            if item.is_symlink() or (hasattr(item, 'is_junction') and item.is_junction()):
                raise ValueError('Artifacts must not contain filesystem links.')
        path = unresolved.resolve()
        if relative.is_absolute() or path == directory or directory not in path.parents or '..' in relative.parts or '\\' in name:
            raise ValueError('Artifact paths must stay inside the bundle.')
        if not path.is_file() or path.is_symlink() or path.stat().st_size != record['bytes'] or file_hash(path) != record['sha256']:
            raise ValueError(f'Artifact is missing or modified: {name}')
    required = {'config.json', 'corpus.jsonl', 'chunks.jsonl', 'dense.index'}
    for model in ('retriever', 'reranker', 'reader'):
        required.update({f'models/{model}/config.json', f'models/{model}/model.safetensors', f'models/{model}/tokenizer_config.json'})
        if not any(f'models/{model}/{name}' in manifest['files'] for name in ('tokenizer.json','vocab.txt','tokenizer.model','sentencepiece.bpe.model')):
            raise ValueError(f'Tokenizer vocabulary is missing for {model}.')
    if not required.issubset(manifest['files']):
        raise ValueError('Artifact manifest omits required runtime files.')
    actual = set()
    for path in directory.rglob('*'):
        if path.is_symlink() or (hasattr(path,'is_junction') and path.is_junction()):
            raise ValueError('Artifacts must not contain filesystem links.')
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != {'manifest.json',*manifest['files']}:
        raise ValueError('The artifact directory differs from its file manifest.')
    return manifest


def write_manifest(directory: Path, provenance: dict) -> dict:
    files = {p.relative_to(directory).as_posix(): {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
             for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    manifest = {'schema_version': 1, 'files': files, 'provenance': provenance}
    (directory / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest


def natural_chunks(corpus: list[dict], tokenizer, max_tokens: int, overlap_tokens: int) -> list[dict]:
    """Create source-aligned token windows, preferring nearby sentence endings."""
    if max_tokens < 32 or not 0 <= overlap_tokens < max_tokens // 2:
        raise ValueError('Require max_tokens >=32 and overlap < half a window.')
    chunks = []
    for doc in corpus:
        text = doc.get('context', doc.get('text'))
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)['offset_mapping']
        offsets = [(a, b) for a, b in offsets if b > a]
        start = 0
        while start < len(offsets):
            end = min(start + max_tokens, len(offsets))
            if end < len(offsets):
                floor = start + max_tokens * 3 // 4
                candidates = [i for i in range(floor, end) if re.search(r'[.!?]\s*$', text[offsets[i][0]:offsets[i][1]+1])]
                if candidates:
                    end = candidates[-1] + 1
            char_start, char_end = offsets[start][0], offsets[end-1][1]
            chunks.append({'chunk_id': len(chunks), 'context_id': doc.get('context_id', doc.get('id')),
                           'title': doc['title'], 'char_start': char_start, 'char_end': char_end,
                           'context': text[char_start:char_end]})
            if end == len(offsets):
                break
            start = max(start + 1, end - overlap_tokens)
    return chunks


def parent_window(text: str, offsets: Sequence[tuple[int, int]], start: int, end: int, budget: int) -> tuple[str, int, int]:
    """Center a bounded source window on retrieval evidence without gold offsets."""
    if budget < 1 or not offsets:
        raise ValueError('A reader window requires tokens and a positive budget.')
    center = (start + end) // 2
    token_center = min(range(len(offsets)), key=lambda i: abs((offsets[i][0] + offsets[i][1]) // 2 - center))
    left = max(0, min(token_center - budget // 2, len(offsets) - budget))
    right = min(len(offsets), left + budget)
    a, b = offsets[left][0], offsets[right - 1][1]
    return text[a:b], a, b


def decode_span(context, offsets, sequence_ids, start_logits, end_logits, cls_index, max_answer_tokens=40, null_score_margin=0.0) -> dict:
    valid = np.asarray([i for i, sid in enumerate(sequence_ids) if sid == 1 and offsets[i][1] > offsets[i][0]], dtype=int)
    if not len(valid):
        return {'answer': '', 'reader_score': 0.0}
    distributions = []
    for values in (start_logits, end_logits):
        masked = np.full(len(values), -np.inf)
        masked[np.append(valid, cls_index)] = np.asarray(values)[np.append(valid, cls_index)]
        weights = np.exp(masked - masked.max())
        distributions.append(weights / weights.sum())
    starts, ends = distributions
    scores = starts[valid, None] * ends[None, valid]
    legal = (valid[None, :] >= valid[:, None]) & (valid[None, :] - valid[:, None] < max_answer_tokens)
    scores[~legal] = -1
    i, j = np.unravel_index(scores.argmax(), scores.shape)
    s, e = int(valid[i]), int(valid[j])
    null_score = float(starts[cls_index] * ends[cls_index])
    score = float(scores[i, j])
    a, b = offsets[s][0], offsets[e][1]
    answer = context[a:b].strip() if score > null_score + null_score_margin else ''
    return {'answer': answer, 'reader_score': score, 'null_score': null_score, 'answer_char_start': a, 'answer_char_end': b}


class DenseEncoder:
    def __init__(self, model, tokenizer, device, config):
        self.model, self.tokenizer, self.device, self.config = model.to(device).eval(), tokenizer, device, config

    def encode(self, texts: Sequence[str], *, queries=False, batch_size=8) -> np.ndarray:
        import torch
        embeddings = []
        prefix = ('query: ' if queries else 'passage: ') if self.config['kind'] == 'e5' else ''
        length = self.config['query_max_length' if queries else 'passage_max_length']
        for start in range(0, len(texts), batch_size):
            encoded = self.tokenizer([prefix + text for text in texts[start:start+batch_size]], padding=True,
                                     truncation=True, max_length=length, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                pooled = mean_pooling(self.model(**inputs), inputs['attention_mask'])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.float().cpu().numpy())
        return np.concatenate(embeddings).astype('float32')


class PortableRAG:
    def __init__(self, bundle: Path, device='auto', *, use_reader=True):
        import faiss
        import torch
        from rank_bm25 import BM25Okapi
        from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
        from transformers.utils import logging
        logging.disable_progress_bar()
        started = time.perf_counter()
        self.bundle = Path(bundle)
        self.manifest = verify_bundle(self.bundle)
        self.config = json.loads((self.bundle/'config.json').read_text(encoding='utf-8'))
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable. Install the CUDA runtime or select CPU.')
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device))
        if self.device.type == 'cpu':
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.corpus = {r['context_id']: r for r in jsonl(self.bundle/'corpus.jsonl')}
        self.chunks = jsonl(self.bundle/'chunks.jsonl')
        self.index = faiss.read_index(str(self.bundle/'dense.index'))
        retriever_dir = self.bundle/'models/retriever'
        self.encoder = DenseEncoder(AutoModel.from_pretrained(retriever_dir, local_files_only=True),
                                    AutoTokenizer.from_pretrained(retriever_dir, local_files_only=True), self.device, self.config['retriever'])
        self.bm25 = BM25Okapi([tokenize_for_bm25(r['context']) for r in self.chunks])
        if self.index.ntotal != len(self.chunks) or self.index.d != self.encoder.model.config.hidden_size:
            raise ValueError('Dense index and model/corpus dimensions differ.')
        for i, row in enumerate(self.chunks):
            if row['chunk_id'] != i or row['context_id'] not in self.corpus:
                raise ValueError('Invalid source identity or chunk order.')
            text = self.corpus[row['context_id']]['context']
            if text[row['char_start']:row['char_end']] != row['context']:
                raise ValueError('Chunk text and source offsets differ.')
        rdir = self.bundle/'models/reranker'
        self.reranker = CrossEncoderReranker(AutoModelForSequenceClassification.from_pretrained(rdir, local_files_only=True),
                                           AutoTokenizer.from_pretrained(rdir, local_files_only=True), self.device, batch_size=8, max_length=512)
        self.reader = None
        self._offsets = {}
        if use_reader:
            rd = self.bundle/'models/reader'
            self.reader = (AutoTokenizer.from_pretrained(rd, local_files_only=True),
                           AutoModelForQuestionAnswering.from_pretrained(rd, local_files_only=True).to(self.device).eval())
        self.load_seconds = time.perf_counter() - started

    def search(self, question: str) -> list[dict]:
        question = validate_question_tokens(question, self.reranker.tokenizer)
        options = self.config['retrieval']
        k = min(options['candidate_k'], len(self.chunks))
        sparse, _ = top_k_from_scores(np.asarray(self.bm25.get_scores(tokenize_for_bm25(question))), k)
        _, dense = self.index.search(self.encoder.encode([question], queries=True), k)
        fused = reciprocal_rank_fusion(sparse, dense[0])[:options['rerank_k']]
        contexts = [self.chunks[i]['context'] for i, _ in fused]
        scores = self.reranker.score([question]*len(contexts), contexts)
        records = [{**self.chunks[i], 'rrf_score': float(rrf), 'reranker_score': float(score)}
                   for (i, rrf), score in zip(fused, scores)]
        records.sort(key=lambda r: r['reranker_score'], reverse=True)
        return [{**r, 'rank': i+1} for i, r in enumerate(records)]

    def read_candidates(self, question: str, evidence: list[dict], reader_config=None) -> list[dict]:
        import torch
        if self.reader is None:
            raise RuntimeError('Reader is disabled.')
        options = reader_config or self.config['reader']
        tokenizer, model = self.reader
        question = validate_question_tokens(question, tokenizer)
        qtokens = len(tokenizer(question, add_special_tokens=False)['input_ids'])
        budget = options.get('max_length', 512) - qtokens - tokenizer.num_special_tokens_to_add(pair=True)
        candidates, seen = [], set()
        for item in evidence[:options['context_k']]:
            context, char_start = item['context'], item['char_start']
            if options['context_mode'] == 'parent_window':
                parent = self.corpus[item['context_id']]['context']
                if item['context_id'] not in self._offsets:
                    self._offsets[item['context_id']] = tokenizer(parent, add_special_tokens=False, return_offsets_mapping=True, verbose=False)['offset_mapping']
                context, char_start, _ = parent_window(parent, self._offsets[item['context_id']], item['char_start'], item['char_end'], budget)
            key = (item['context_id'], char_start, context)
            if key in seen:
                continue
            seen.add(key)
            encoded = tokenizer(question, context, truncation='only_second', max_length=options.get('max_length', 512),
                                return_offsets_mapping=True, return_tensors='pt')
            offsets = encoded.pop('offset_mapping')[0].tolist()
            sequences = encoded.sequence_ids(0)
            cls = encoded['input_ids'][0].tolist().index(tokenizer.cls_token_id)
            with torch.inference_mode():
                result = model(**{k:v.to(self.device) for k,v in encoded.items()})
            span = decode_span(context, offsets, sequences, result.start_logits[0].float().cpu().numpy(), result.end_logits[0].float().cpu().numpy(), cls,
                               null_score_margin=options.get('null_score_margin',0.0))
            candidates.append({**item, **span, 'context': context, 'reader_window_start': char_start})
        return candidates

    @staticmethod
    def select_answer(question, evidence, candidates, thresholds, reader_config):
        accepted = []
        for item in candidates:
            if item.get('null_score') is not None and item['reader_score'] <= item['null_score'] + reader_config.get('null_score_margin',0.0):
                continue
            if item['rrf_score'] < thresholds['min_rrf_score'] or item['reranker_score'] < thresholds['min_reranker_score'] or item['reader_score'] < thresholds['min_reader_score']:
                continue
            if thresholds.get('require_explicit_year', True):
                years = set(re.findall(r'(?<!\d)(\d{4})\s*년', question))
                if not years.issubset(set(re.findall(r'(?<!\d)(\d{4})\s*년', item['context']))):
                    continue
            answer = item['answer']
            if answer and ''.join(answer.lower().split()) in ''.join(item['context'].lower().split()):
                accepted.append(item)
        if not accepted:
            return {'answer':'', 'answered':False, 'answer_source':'abstained', 'evidence':evidence}
        def score(item):
            return item['reader_score'] * (item['reranker_score'] if reader_config.get('score_mode') == 'reader_reranker' else 1.)
        best = max(accepted, key=lambda r:(score(r), -r['rank']))
        return {'answer':best['answer'], 'answered':True, 'answer_source':'extractive_reader',
                'reader_score':best['reader_score'], 'reranker_score':best['reranker_score'], 'answer_in_evidence':True,
                'citation':{k:best[k] for k in ('title','chunk_id','context_id','context','reader_window_start','answer_char_start','answer_char_end')}, 'evidence':evidence}

    def answer(self, question):
        evidence = self.search(question)
        if self.reader is None:
            return {'answer':None, 'answer_source':'retrieval_only', 'evidence':evidence}
        candidates = self.read_candidates(question, evidence)
        return self.select_answer(question, evidence, candidates, self.config['thresholds'], self.config['reader'])
