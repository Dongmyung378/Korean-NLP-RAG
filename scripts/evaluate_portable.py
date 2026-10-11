"""Evaluate immutable development/final splits with auditable reader caches."""
from __future__ import annotations
import argparse, importlib.metadata, json, platform, sys, time
from pathlib import Path, PurePosixPath
if __package__ in {None,''}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from korean_rag.runtime import PortableRAG, file_hash
from korean_rag.benchmark import read_jsonl, verify_benchmark_directory, score_predictions, relevant_chunk_ids, retrieval_metrics


REQUIRED_FINAL_SOURCES = frozenset({
    'korean_rag/runtime.py', 'korean_rag/retrieval.py', 'korean_rag/reranking.py',
    'korean_rag/validation.py', 'korean_rag/benchmark.py', 'korean_rag/qa_metrics.py',
    'korean_rag/abstention_benchmark.py', 'scripts/evaluate_portable.py',
})


def validate_relative_hash_paths(hashes, label):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f'{label} must be a nonempty hash mapping.')
    for name in hashes:
        if (not isinstance(name, str) or not name or '\\' in name or ':' in name
                or PurePosixPath(name).is_absolute()
                or any(part in {'', '.', '..'} for part in name.split('/'))):
            raise ValueError(f'{label} paths must be canonical forward-slash relative paths.')


def verify_frozen_final(frozen, *, configuration, input_metadata, benchmark_manifest,
                        project_root, bundle_directory):
    """Require a complete dependency binding before final questions are read."""
    if configuration.get('evaluation_binding') != input_metadata:
        raise ValueError('Final data hashes must be bound in the frozen bundle configuration.')
    source_hashes = frozen.get('source_sha256')
    artifact_hashes = frozen.get('artifact_sha256')
    validate_relative_hash_paths(source_hashes, 'Source')
    validate_relative_hash_paths(artifact_hashes, 'Artifact')
    if not REQUIRED_FINAL_SOURCES.issubset(source_hashes):
        raise ValueError('The final freeze omits inference or metric source dependencies.')
    if 'manifest.json' not in artifact_hashes:
        raise ValueError('The final freeze must bind the complete verified bundle manifest.')
    artifact_manifest = json.loads((bundle_directory/'manifest.json').read_text(encoding='utf-8'))
    if not set(artifact_hashes).issubset({'manifest.json', *artifact_manifest['files']}):
        raise ValueError('Frozen artifacts must belong to the verified bundle manifest.')
    sources = {name: file_hash(project_root/name) for name in source_hashes}
    artifacts = {name: file_hash(bundle_directory/name) for name in artifact_hashes}
    from korean_rag.benchmark import verify_final_freeze
    verify_final_freeze(frozen, benchmark_sha256=benchmark_manifest['benchmark_sha256'],
                       protocol_sha256=benchmark_manifest['protocol_sha256'],
                       configuration=configuration, source_sha256=sources, artifact_sha256=artifacts)


def environment_metadata(pipeline):
    """Describe the actual execution environment without claiming bitwise parity."""
    import torch
    packages = ('numpy', 'torch', 'transformers', 'tokenizers', 'faiss-cpu',
                'rank-bm25', 'safetensors', 'huggingface-hub')
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    device = torch.device(pipeline.device)
    gpu_index = (device.index if device.index is not None else torch.cuda.current_device()) if device.type == 'cuda' else None
    return {
        'python_version': platform.python_version(), 'python_implementation': platform.python_implementation(),
        'os': platform.system(), 'os_release': platform.release(), 'machine': platform.machine(),
        'dependency_versions': versions, 'device': str(device), 'gpu_index': gpu_index,
        'gpu_name': torch.cuda.get_device_name(gpu_index) if gpu_index is not None else None,
        'gpu_compute_capability': list(torch.cuda.get_device_capability(gpu_index)) if gpu_index is not None else None,
        'torch_version': str(torch.__version__), 'cuda_runtime_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'torch_num_threads': torch.get_num_threads(), 'torch_num_interop_threads': torch.get_num_interop_threads(),
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'deterministic_algorithms_enabled': torch.are_deterministic_algorithms_enabled(),
        'numerical_scope': 'Hashes reproduce data, source and artifacts. Hardware and backend arithmetic can differ; fixed near-null guards do not guarantee universal bitwise equality.',
    }


def evaluate(pipeline, questions, output, cache_reader_variants=False, input_metadata=None):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite evidence: {output}')
    output.mkdir(parents=True)
    project_root=Path(__file__).resolve().parents[1]
    source_hashes={name:file_hash(project_root/name) for name in sorted(REQUIRED_FINAL_SOURCES)}
    environment=environment_metadata(pipeline)
    predictions, rankings, latencies = [], [], []
    variants = [
        {'context_k':3,'context_mode':'chunk','max_length':512,'score_mode':'reader'},
        {'context_k':3,'context_mode':'parent_window','max_length':512,'score_mode':'reader'},
        {'context_k':5,'context_mode':'parent_window','max_length':512,'score_mode':'reader'},
        {'context_k':5,'context_mode':'parent_window','max_length':512,'score_mode':'reader_reranker'},
    ]
    with (output/'reader-cache.jsonl').open('x',encoding='utf-8') as cache:
        for i, q in enumerate(questions):
            started=time.perf_counter()
            evidence=pipeline.search(q['question'])
            rankings.append([r['chunk_id'] for r in evidence])
            if cache_reader_variants:
                chunk=pipeline.read_candidates(q['question'],evidence,variants[0])
                parent=pipeline.read_candidates(q['question'],evidence,variants[2])
                pools={'chunk':chunk,'parent_window':parent}
                candidates=[r for r in pools[pipeline.config['reader']['context_mode']] if r['rank']<=pipeline.config['reader']['context_k']]
            else:
                candidates=pipeline.read_candidates(q['question'],evidence)
                pools={pipeline.config['reader']['context_mode']:candidates}
            result=pipeline.select_answer(q['question'],evidence,candidates,pipeline.config['thresholds'],pipeline.config['reader'])
            latencies.append(time.perf_counter()-started)
            predictions.append({'query_id':q['query_id'],**result})
            cache.write(json.dumps({'query_id':q['query_id'],'question':q['question'],'evidence':evidence,'pools':pools},ensure_ascii=False)+'\n')
            if (i+1)%25==0:
                print(json.dumps({'completed':i+1,'total':len(questions)}),flush=True)
    with (output/'predictions.jsonl').open('x',encoding='utf-8') as f:
        for row in predictions:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary, _ = score_predictions(questions,predictions)
    import numpy as np
    answerable=[q for q in questions if q.get('answerable',True)]
    gold=[relevant_chunk_ids(q,pipeline.chunks) for q in answerable]
    qranks=[r for q,r in zip(questions,rankings) if q.get('answerable',True)]
    retrieval=retrieval_metrics(qranks,gold) if gold else None
    if source_hashes != {name:file_hash(project_root/name) for name in sorted(REQUIRED_FINAL_SOURCES)}:
        raise RuntimeError('Evaluation source changed during execution; preserve partial evidence and rerun as an execution failure.')
    report={'configuration':pipeline.config,'bundle_manifest_sha256':file_hash(pipeline.bundle/'manifest.json'),
            'environment':environment, 'source_sha256':source_hashes,
            'qa':summary,'retrieval':retrieval,'load_seconds':pipeline.load_seconds,
            'latency_mean_seconds':float(np.mean(latencies)), 'latency_p50_seconds':float(np.median(latencies)),
            'latency_p95_seconds':float(np.quantile(latencies,.95)),
            'latency_scope':'Includes both reader contexts when cache_reader_variants=true; final serving latency is measured separately.' if cache_reader_variants else 'Single configured serving path, model loading excluded.',
            'cache_reader_variants':cache_reader_variants, 'inputs':input_metadata or {},
            'reader_cache_sha256':file_hash(output/'reader-cache.jsonl'),
            'predictions_sha256':file_hash(output/'predictions.jsonl')}
    (output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--split',choices=['dev','final'],default='dev')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    parser.add_argument('--cache-reader-variants',action='store_true')
    parser.add_argument('--no-answer',type=Path)
    parser.add_argument('--freeze',type=Path)
    args=parser.parse_args()
    manifest=verify_benchmark_directory(args.benchmark,Path(__file__).resolve().parents[1]/'configs/validation-protocol.json')
    input_metadata={'split':args.split,'benchmark_sha256':manifest['benchmark_sha256'],
                    'question_file_sha256':file_hash(args.benchmark/f'{args.split}.jsonl')}
    if args.no_answer:
        from korean_rag.abstention_benchmark import verify_abstention_directory
        abstention=verify_abstention_directory(args.no_answer.parent,Path(__file__).resolve().parents[1]/'configs/no-answer-protocol.json')
        if args.no_answer.name!=f'{args.split}.jsonl' or abstention['answerable_benchmark_sha256']!=manifest['benchmark_sha256']:
            parser.error('No-answer inputs must match the verified question split and corpus.')
        input_metadata.update(no_answer_benchmark_sha256=abstention['no_answer_benchmark_sha256'],no_answer_file_sha256=file_hash(args.no_answer))
    if args.split=='final':
        if args.cache_reader_variants or not args.freeze:
            parser.error('Final evaluation requires a frozen configuration and forbids variant caches.')
        frozen=json.loads(args.freeze.read_text(encoding='utf-8'))
        from korean_rag.runtime import verify_bundle
        verify_bundle(args.bundle)
        configuration=json.loads((args.bundle/'config.json').read_text(encoding='utf-8'))
        root=Path(__file__).resolve().parents[1]
        verify_frozen_final(frozen,configuration=configuration,input_metadata=input_metadata,
                            benchmark_manifest=manifest,project_root=root,bundle_directory=args.bundle)
    pipeline=PortableRAG(args.bundle,args.device)
    questions=read_jsonl(args.benchmark/f'{args.split}.jsonl')
    if args.no_answer:
        questions+=read_jsonl(args.no_answer)
    evaluate(pipeline,questions,args.output,args.cache_reader_variants,input_metadata)


if __name__=='__main__':
    main()
