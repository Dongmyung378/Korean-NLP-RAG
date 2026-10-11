"""Prepare auditable synthetic no-answer dev/final files without inference."""

import argparse
import json
from pathlib import Path

from korean_rag.abstention_benchmark import prepare_abstention_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/no-answer-protocol.json"))
    parser.add_argument("--regression-file", type=Path, default=Path("tests/data/unanswerable_questions.json"))
    parser.add_argument("--regression-source-review", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_abstention_benchmark(benchmark_dir=args.benchmark_dir,
        protocol_path=args.protocol, regression_path=args.regression_file,
        output_dir=args.output_dir, review_path=args.regression_source_review)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
