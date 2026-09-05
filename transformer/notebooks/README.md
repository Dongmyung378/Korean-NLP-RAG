# 노트북 안내

4주간의 학습·검색·질의응답 실험 기록입니다. 노트북은 Kaggle에서 독립 실행하고, 로컬 앱은 저장된 모델과 인덱스만 불러옵니다.

## 주차별 역할

| 주차 | 노트북 | 주요 실험 | 최종 앱과의 관계 |
| --- | --- | --- | --- |
| 1 | [Transformer fine-tuning](<Week - 1/korean-transformer-finetuning.ipynb>) | KLUE-YNAT 뉴스 분류, 설정 비교, hold-out 평가 | 학습·평가 실험이며 분류 가중치는 재사용하지 않음 |
| 2 | [Sentence embedding](<Week - 2/sentence-embedding-retrieval.ipynb>) | KLUE-STS 학습, 다중 시드, FAISS 검색 | Dense Retriever와 토크나이저 |
| 3 | [Cross-Encoder reranker](<Week - 3/cross-encoder-reranker.ipynb>) | hard-negative 학습, 검색 순위 평가 | Reranker 체크포인트 |
| 4 | [Hybrid RAG](<Week - 4/hybrid-rag.ipynb>) | BM25·Dense·RRF·리랭킹, Qwen·Reader 비교 | 지식베이스, 검색 인덱스, 답변 평가 |

주차별 요약: [1주차](../../reports/week1_experiment_results.md) · [2주차](../../reports/week2_experiment_results.md) · [3주차](../../reports/week3_experiment_results.md) · [4주차](../../reports/week4_experiment_results.md)

## Kaggle 재현 순서

1. 각 노트북을 새 세션에서 위에서 아래로 실행합니다. 모델·데이터 다운로드를 위한 Internet과 학습용 GPU를 준비합니다.
2. 3주차 입력으로 2주차 출력물을 연결합니다.
3. 4주차 입력으로 2주차 Retriever와 3주차 Reranker 출력물을 연결합니다.
4. 4주차 완료 후 `week4_final_artifacts.zip`을 받아 압축을 풉니다.

4주차 입력 셀의 `WEEK2_CHECKPOINT`, `WEEK3_MODEL_DIR`는 기본값이 `None`입니다. 입력에 모델이 하나씩 있으면 자동으로 찾고, 여러 버전이 있으면 실행을 멈춥니다. 그때는 두 변수에 사용할 체크포인트 파일과 리랭커 폴더의 전체 Kaggle 경로를 지정하세요. 2주차 설정과 토크나이저도 선택한 체크포인트 옆에서만 읽습니다. 입력 검사만 추가했으며 기존 모델과 출력물은 그대로입니다.

로컬 실행만 하려면 이 과정을 반복할 필요가 없습니다. [Google Drive의 outputs](https://drive.google.com/drive/folders/1wqPSshOLzAJHYGfR7Z-Ug0BP_k3ycL4s?usp=sharing)를 프로젝트의 `transformer/outputs`에 배치하고 [최상위 README](../../README.md)의 실행 방법을 따르면 됩니다.

## 4주차 결과와 실행 기록

수정본은 Kaggle 전체 실행을 완료했습니다. 고정 질문 100개에서 최종 정답 포함은 74개였고, Reader 후보 209개에 빈 답 안내 문장은 없습니다. 모든 최종 답변은 Reader에서 나왔으며 이 표본에서는 Qwen fallback이 사용되지 않았습니다.

이후 한 글자 답변의 인용 누락을 로컬에서 수정했습니다. 저장된 답변은 바꾸지 않고 인용·표·그림만 갱신했으며, 해당 셀에 `local_postprocessing` 메타데이터를 남겼습니다. 모델 추론 시간은 원래 Kaggle 값을 유지했습니다. [후처리 기록](../../reports/week4_postprocessing.json)에서 변경 범위를 확인할 수 있습니다.

현재 노트북 결과를 단순히 수정 전 Kaggle 출력으로만 보아서는 안 됩니다. 학습·추론은 Kaggle 실행, 인용·보고서 일부는 저장 결과의 로컬 후처리로 구분돼 있습니다.

로컬 앱은 Qwen 없이 후보별 점수와 연도 검사를 사용하므로 정답 포함은 67/100입니다. 노트북의 74/100과 같은 정책의 성능으로 합쳐 표기하지 않습니다.

## 결과를 해석할 때

- 지식베이스는 KorQuAD 1.0 validation split의 한국어 위키백과 문맥이며 웹 전체를 검색하지 않습니다.
- 정답 포함률은 정규화된 정답 문자열의 포함 여부입니다. 인용률 역시 의미적 정답률이 아닙니다.
- 4주차 질문은 기존 개발에 사용한 표본입니다. 일반적인 무응답 성능이나 새 held-out 성능을 주장하지 않습니다.
- 2주차 배포 JSON의 `sts_baseline`은 이름과 달리 2일차 학습 결과를 담습니다. 사전학습 기준선과 최종 시드 결과는 [2주차 보고서](../../reports/week2_experiment_results.md)에 구분했습니다.
- 3주차 random-negative 기준 모델은 4일차 검증에 포함된 질문 일부를 이미 학습했습니다. 기준선과 hard-negative 모델의 점수 차이를 공정한 효과 비교로 해석하지 않습니다. 이번에는 [설명을 바로잡았으며](../../reports/week3_experiment_results.md), 비교 모델을 다시 학습한 것은 아닙니다.

## 선택 사항: 보고서 재생성

일반적인 로컬 질의응답에는 필요하지 않습니다. 저장된 답변과 후보로 인용·표·그림을 다시 만들 때만 프로젝트 루트에서 실행합니다.

```bash
python -m pip install -r requirements-reports.txt
python scripts/refresh_week4_reports.py
```

이 명령은 모델 추론 없이 4주차 결과 파일, 노트북의 후처리 출력, 후처리 기록을 갱신합니다. 원본 실행 기록을 보관하려면 먼저 노트북과 출력물의 별도 사본을 보관하세요. 전체 노트북 재학습에 필요한 라이브러리는 각 노트북의 설치·가져오기 셀을 확인해야 하며, 로컬 앱용 `requirements.txt`만으로 학습 환경 전체를 설치하는 것은 아닙니다.
