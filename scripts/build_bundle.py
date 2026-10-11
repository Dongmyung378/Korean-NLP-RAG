"""Build one portable retrieval index from source-aligned chunks and HF models."""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from korean_rag.runtime import DenseEncoder, natural_chunks, jsonl, write_manifest, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--retriever', type=Path, required=True)
    parser.add_argument('--reranker', type=Path, required=True)
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--legacy-chunks', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing bundle.')
    import faiss, torch
    from transformers import AutoTokenizer, AutoModel
    config = json.loads(args.config.read_text(encoding='utf-8'))
    corpus = jsonl(args.corpus)
    tokenizer = AutoTokenizer.from_pretrained(args.retriever, local_files_only=True)
    if args.legacy_chunks:
        import csv
        keys = {doc['context_key']: doc['context_id'] for doc in corpus}
        with args.legacy_chunks.open(encoding='utf-8-sig',newline='') as f:
            chunks = [{'chunk_id':int(r['chunk_id']), 'context_id':keys[r['context_key']], 'title':r['title'],
                       'char_start':int(r['char_start']), 'char_end':int(r['char_end']), 'context':r['chunk_text']} for r in csv.DictReader(f)]
    else:
        chunks = natural_chunks(corpus, tokenizer, **config['chunking'])
    device = torch.device(args.device)
    encoder = DenseEncoder(AutoModel.from_pretrained(args.retriever,local_files_only=True), tokenizer, device, config['retriever'])
    vectors = encoder.encode([r['context'] for r in chunks],batch_size=8)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    args.output.mkdir(parents=True)
    for kind, source in [('retriever',args.retriever),('reranker',args.reranker),('reader',args.reader)]:
        destination = args.output/'models'/kind
        destination.mkdir(parents=True)
        for file in source.iterdir():
            if file.is_file() and file.suffix in {'.json','.safetensors','.txt'} and not file.name.endswith('manifest.json') and 'report' not in file.name:
                shutil.copyfile(file, destination/file.name)
    shutil.copyfile(args.corpus,args.output/'corpus.jsonl')
    (args.output/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    with (args.output/'chunks.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for row in chunks:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    faiss.write_index(index,str(args.output/'dense.index'))
    provenance = {'corpus_sha256':file_hash(args.corpus),'build_config_sha256':file_hash(args.config),
                  'source_sha256':file_hash(Path(__file__).resolve()), 'chunk_count':len(chunks),
                  'index_metric':'inner product over L2-normalized masked mean embeddings',
                  'annotation_independence':'Corpus chunking uses source text only, never answer offsets.'}
    if args.legacy_chunks:
        provenance['legacy_chunks_sha256']=file_hash(args.legacy_chunks)
    write_manifest(args.output,provenance)
    print(json.dumps({'bundle':str(args.output),'chunks':len(chunks),'dimension':index.d}),flush=True)


if __name__=='__main__':
    main()
