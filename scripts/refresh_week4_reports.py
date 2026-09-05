"""저장된 Reader 답변은 유지하고 인용과 결과표만 갱신한다. 모델 추론은 하지 않는다."""

import ast
import base64
import contextlib
import hashlib
import io
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "transformer/outputs/week4/week4_day1_3_artifacts"
NOTEBOOK = ROOT / "transformer/notebooks/Week - 4/hybrid-rag.ipynb"


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    examples_path = ARTIFACTS / "week4_day5_rag_examples.csv"
    original_hash = hashlib.sha256(examples_path.read_bytes()).hexdigest()
    examples = pd.read_csv(examples_path).fillna("")
    original_answers = examples.generated_answer.copy()
    original_grounding = float(examples.answer_is_grounded.mean())
    candidates = pd.read_csv(ARTIFACTS / "week4_day5_reader_candidates.csv")
    chunks = pd.read_csv(ARTIFACTS / "week4_knowledge_base_chunks.csv")
    queries = pd.read_csv(ARTIFACTS / "week4_evaluation_queries.csv")
    rankings = pd.read_csv(ARTIFACTS / "week4_day4_reranked_top20_candidates.csv")
    metrics = json.loads((ARTIFACTS / "week4_day5_generation_metrics.json").read_text())
    base = json.loads((ARTIFACTS / "week4_day1_3_metrics.json").read_text())
    reranking = json.loads((ARTIFACTS / "week4_day4_reranker_metrics.json").read_text())
    strategy = pd.read_csv(ARTIFACTS / "week4_day5_generation_strategy_examples.csv").fillna("")
    summary = pd.read_csv(ARTIFACTS / "week4_day6_generation_strategy_summary.csv")
    efficiency = pd.read_csv(ARTIFACTS / "week4_day6_model_efficiency.csv")
    top_candidates = candidates.sort_values("candidate_rank").groupby("query_id").first()
    scope = {"np": np, "pd": pd, "re": re, "json": json, "plt": plt, "chunk_df": chunks}
    # Qwen 생성 함수는 실행하지 않고 평가용 순수 함수만 가져온다.
    needed = {"normalize_text", "compact_match_text", "answer_is_grounded", "build_answer_citations", "evaluate_rankings"}
    for source in sources:
        if source.startswith("# 라이브러리 설치"):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in needed]
        exec(compile(ast.Module(body=functions, type_ignores=[]), "notebook_helpers", "exec"), scope)
    for index, row in examples.iterrows():
        if row.answer_source != "extractive_reader":
            continue
        chosen = top_candidates.loc[row.query_id]
        if str(chosen.answer) != str(row.generated_answer):
            raise ValueError(f"Saved answer does not match the top Reader candidate: {row.query_id}")
        chunk_ids = [int(chosen.chunk_id)]
        citations = scope["build_answer_citations"](row.generated_answer, chunk_ids, allow_short=True)
        examples.at[index, "citations"] = json.dumps(citations, ensure_ascii=False)
        examples.at[index, "citation_count"] = len(citations)
        examples.at[index, "answer_is_grounded"] = scope["answer_is_grounded"](row.generated_answer, chunk_ids, allow_short=True)
    if "qwen_correction_used" not in examples:
        examples["qwen_correction_used"] = examples.correction_used
    examples.loc[examples.answer_source == "extractive_reader", "correction_used"] = False
    assert original_answers.equals(examples.generated_answer)
    mask = strategy.strategy.isin(["Extractive QA Reader", "Reader-First Final Answer"])
    by_id = examples.set_index("query_id")
    for index, row in strategy.loc[mask].iterrows():
        if row.answer_source == "extractive_reader":
            strategy.at[index, "answer_is_grounded"] = bool(by_id.loc[row.query_id, "answer_is_grounded"])
    for index, row in summary.iterrows():
        selected = strategy.loc[strategy.strategy == row.strategy]
        summary.at[index, "grounded_answer_rate"] = selected.answer_is_grounded.mean()
    metrics.update({
        "qwen_validation_correction_rate": metrics.get("qwen_validation_correction_rate", metrics["correction_rate"]),
        "correction_rate": float(examples.correction_used.mean()),
        "grounded_answer_rate": float(examples.answer_is_grounded.mean()),
        "citation_coverage": float((examples.citation_count > 0).mean()),
        "postprocessing_note": "Reader citations refreshed from saved candidates; answers and Kaggle inference timings unchanged.",
    })
    grouped = rankings.sort_values("rerank_rank").groupby("query_id")["chunk_id"].apply(list)
    indices = np.asarray([grouped.loc[qid] for qid in queries.query_id], dtype=np.int64)
    scope.update({
        "reranked_indices": indices, "relevant_sets": [set(json.loads(value)) for value in queries.gold_chunk_ids],
        "RAG_CONTEXT_K": 3, "bm25_metrics": base["bm25"], "dense_metrics": base["dense"],
        "hybrid_metrics": base["hybrid_rrf"], "reranked_metrics": reranking["hybrid_reranked"],
        "generation_strategy_summary_df": summary, "generation_metrics": metrics,
        "TARGET_ANSWER_COVERAGE": 0.6, "rag_generation_df": examples,
        "strategy_latency_seconds": dict(zip(summary.strategy, summary.seconds_per_query)),
        "bm25_latency_ms": base["latency_ms_per_query"]["bm25"],
        "dense_encoding_ms": base["latency_ms_per_query"]["dense_encoding"],
        "dense_search_ms": base["latency_ms_per_query"]["dense_search"],
        "hybrid_fusion_ms": base["latency_ms_per_query"]["rrf_fusion"],
        "reranker_latency_ms_per_query": reranking["reranker_latency_ms_per_query"],
        "reranker_peak_memory_mb": reranking["reranker_peak_gpu_memory_mb"],
        "week2_parameter_count": int(efficiency.iloc[0].parameter_count),
        "reranker_parameter_count": reranking["reranker_parameters"],
        "generator_parameter_count": metrics["generator_parameters"],
        "reader_parameter_count": metrics["reader_parameters"],
        "EXTRACTIVE_READER_MODEL_NAME": metrics["extractive_reader_model"], "SEED": base["seed"],
    })
    outputs = []

    def display(frame):
        outputs.append({"output_type": "display_data", "metadata": {}, "data": {
            "text/plain": frame.to_string(index=False), "text/html": frame.to_html(index=False),
        }})

    def show():
        buffer = io.BytesIO()
        plt.gcf().savefig(buffer, format="png", bbox_inches="tight")
        outputs.append({"output_type": "display_data", "metadata": {}, "data": {
            "image/png": base64.b64encode(buffer.getvalue()).decode("ascii"), "text/plain": "Locally refreshed figure",
        }})
        plt.close()

    scope["display"] = display
    plt.show = show
    stamp = datetime.now(timezone.utc).isoformat()
    note = "Local postprocessing: cached Reader answers and original inference timings retained; citations and tables refreshed.\n"
    with TemporaryDirectory(prefix="week4-report-") as temporary:
        target = Path(temporary)
        scope["artifact_dir"] = target
        display(summary)
        display(examples[["question", "gold_answer", "reader_answer", "generated_answer", "answer_source", "gold_answer_in_generated_answer", "citations"]].head(10))
        notebook["cells"][39]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": note}, *outputs]
        for index, source in [(42, sources[42]), (45, sources[45].split("final_manifest =")[0])]:
            outputs.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, f"notebook_cell_{index}", "exec"), scope)
            notebook["cells"][index]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": note}, *outputs]
        for index in (39, 42, 45):
            notebook["cells"][index]["metadata"]["local_postprocessing"] = {"at_utc": stamp, "model_inference": False}
        examples.drop(columns=["error_category"], errors="ignore").to_csv(target / examples_path.name, index=False, encoding="utf-8-sig")
        strategy.to_csv(target / "week4_day5_generation_strategy_examples.csv", index=False, encoding="utf-8-sig")
        (target / "week4_day5_generation_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for file in target.iterdir():
            shutil.copy2(file, ARTIFACTS / file.name)
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    audit = {
        "updated_at_utc": stamp, "model_inference": False, "answers_changed": 0,
        "answer_containment_rate": float(examples.gold_answer_in_generated_answer.mean()),
        "grounded_rate_before": original_grounding, "grounded_rate_after": metrics["grounded_answer_rate"],
        "input_examples_sha256": original_hash, "output_examples_sha256": hashlib.sha256(examples_path.read_bytes()).hexdigest(),
        "timing_note": "Original Kaggle component timings retained, not a new end-to-end latency measurement.",
    }
    (ROOT / "reports/week4_postprocessing.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
