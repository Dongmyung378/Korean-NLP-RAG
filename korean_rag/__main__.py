"""Single-question, persistent interactive and JSONL Korean QA commands."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .validation import QuestionValidationError, validate_question_text

def _print_result(result):
    answer=result.get('answer')
    print(f"Answer: {answer or ('Evidence unavailable for this question.' if result['answer_source']=='abstained' else 'Retrieval only.')}")
    print(f"Source: {result['answer_source']}")
    if 'citation' in result:
        citation=result['citation']
        print(f"Citation: {citation['title']} (chunk {citation['chunk_id']})")
        print(citation['context'])
    else:
        for record in result.get('evidence',[])[:3]:
            print(f"{record['rank']}. {record['title']}: {record['context'][:220]}")

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--question','-q')
    mode.add_argument('--interactive',action='store_true',help='Keep loaded models for multiple questions; exit with /quit or EOF.')
    mode.add_argument('--questions-file',type=Path,help='UTF-8 file with one question per line; output one JSON object per line.')
    parser.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parents[1]/'artifacts/runtime')
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--no-reader',action='store_true')
    parser.add_argument('--outputs-root',type=Path,help='Opt-in historical Week2-4 inference for regression comparisons.')
    args=parser.parse_args(argv)
    questions=None
    if args.questions_file:
        try:
            questions=args.questions_file.read_text(encoding='utf-8-sig').splitlines()
        except OSError as error:
            parser.error(str(error))
    elif args.question is not None:
        try:
            questions=[validate_question_text(args.question)]
        except ValueError as error:
            parser.error(str(error))
    elif not args.interactive:
        try:
            questions=[validate_question_text(input('Question: '))]
        except (ValueError,EOFError) as error:
            parser.error(str(error) or 'No question was supplied.')
    try:
        if args.outputs_root:
            from .pipeline import KoreanRAGPipeline
            from .retrieval import ArtifactPaths
            pipeline=KoreanRAGPipeline(ArtifactPaths(args.outputs_root),device=args.device,use_reader=not args.no_reader)
        else:
            from .runtime import PortableRAG
            pipeline=PortableRAG(args.bundle,device=args.device,use_reader=not args.no_reader)
    except (FileNotFoundError,ValueError,RuntimeError,ImportError) as error:
        print(f'Runtime preparation failed: {error}',file=sys.stderr)
        print('Run the project launcher or scripts/doctor.py to check the environment and artifacts.',file=sys.stderr)
        return 1
    def process(question):
        try:
            question=validate_question_text(question)
            result=pipeline.answer(question)
        except QuestionValidationError as error:
            result={'question':question,'error':str(error),'answer_source':'invalid_input'}
        if args.json or args.questions_file:
            print(json.dumps({'question':question,**result},ensure_ascii=False),flush=True)
        elif 'error' in result:
            print(result['error'],file=sys.stderr)
        else:
            _print_result(result)
    if questions is not None:
        for question in questions:
            if question.strip():
                process(question)
        return 0
    print('Models ready. Enter one question per line; /quit exits.',file=sys.stderr)
    while True:
        try:
            line=input()
        except (EOFError,KeyboardInterrupt):
            return 0
        if line.strip()=='/quit':
            return 0
        if line.strip():
            process(line)

if __name__=='__main__':
    raise SystemExit(main())
