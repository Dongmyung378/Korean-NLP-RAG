# Korean NLP RAG

한국어 Transformer 학습부터 문장 임베딩, 검색, 리랭킹, 근거 기반 질의응답까지 구현한 4주 프로젝트입니다. 노트북에서 모델을 학습·평가하고, 저장한 모델과 검색 인덱스를 Python 모듈로 불러와 터미널에서 질문에 답합니다.

## 구현 범위

- 1주차: KLUE-YNAT 뉴스 제목 분류와 학습 설정 비교
- 2주차: KLUE-STS 문장 임베딩 학습, 다중 시드 비교, FAISS 검색
- 3주차: hard-negative Cross-Encoder 리랭커 학습과 검색 순위 평가
- 4주차: BM25 + Dense Retrieval 결합, Qwen·추출형 Reader 비교, 인용과 오류 분석
- 로컬 앱: 검색부터 답변 추출·근거 인용·답변 보류까지 CLI로 실행

1주차 분류 모델은 최종 검색기에 직접 연결하지 않습니다. 로컬 앱은 2주차 Retriever, 3주차 Reranker, 4주차 지식베이스와 인덱스를 사용합니다.

`질문 -> BM25 Top-50 + Dense Top-50 -> RRF Top-20 -> Cross-Encoder Top-3 -> QA Reader -> 답변·인용 또는 보류`

노트북은 Qwen 생성 모델과 Reader-First 전략을 비교합니다. 로컬 앱은 Qwen 없이 추출형 Reader와 후보별 점수 기준을 사용합니다.

## 어떤 질문에 답할 수 있나요?

지식베이스는 **KorQuAD 1.0 validation split의 한국어 위키백과 문맥** 960개를 분할한 5,889개 청크입니다. 이 문맥에 포함된 인물·문화·역사·과학 등의 사실을 검색합니다. 웹 검색이나 사용자 문서 업로드 기능은 없습니다.

실시간 날씨, 최신 뉴스, 개인 정보, 저장된 문맥 밖의 지식에는 신뢰할 만한 답을 제공할 수 없습니다. 관련 문맥이 있더라도 질문의 전제가 틀리거나 검색·추출에 실패하면 오답이 나올 수 있습니다. 답변과 함께 인용 근거를 확인해야 합니다.

## 빠른 실행

Python 3.12 환경에서 프로젝트 최상위 폴더를 기준으로 실행합니다.

### 1. 저장소와 모델 준비

```bash
git clone https://github.com/Dongmyung378/Korean-NLP-RAG.git
cd Korean-NLP-RAG
```

이미 저장소를 받았다면 해당 폴더로 이동하면 됩니다.

학습 체크포인트는 각각 100MB를 넘어 GitHub에 포함하지 않습니다. [Google Drive에서 outputs 다운로드](https://drive.google.com/drive/folders/1wqPSshOLzAJHYGfR7Z-Ug0BP_k3ycL4s?usp=sharing) 후 다음 위치에 넣어 주세요. ZIP으로 받았다면 압축을 풀어야 합니다.

```text
Korean-NLP-RAG/
  korean_rag/
  transformer/
    notebooks/
    outputs/
      week2/
        week2_retriever_artifacts/
          week2_dense_retriever.pt
          tokenizer/
      week3/
        week3_day4_final_reranker/
          config.json
          model.safetensors
          tokenizer.json
          tokenizer_config.json
      week4/
        week4_day1_3_artifacts/
          week4_knowledge_base_chunks.csv
          week4_dense.index
          week4_bm25.pkl
          week4_day5_rag_examples.csv
```

위 구조는 주요 파일만 표시한 것입니다. 개별 파일을 골라 받기보다 배포된 폴더 구조를 유지하세요. `outputs/outputs/`처럼 중복 폴더가 생기지 않도록 확인해 주세요. 1주차 출력물은 로컬 질의응답 실행에 필요하지 않습니다.

체크포인트와 BM25 파일은 역직렬화해서 읽으므로 신뢰할 수 있는 출처의 파일만 사용하세요.

### 2. 의존성 설치

```bash
python -m pip install -r requirements.txt
```

가상환경 사용을 권장합니다. 이미 실행되는 환경이 있다면 그대로 사용할 수 있습니다. 첫 답변 실행 때 공개 QA 모델 `monologg/koelectra-base-v3-finetuned-korquad`를 다운로드하며, 이후에는 로컬 캐시를 사용합니다.

검증 환경은 Windows, Python 3.12.7, PyTorch 2.5.1+cu121, Transformers 5.16.1입니다. CPU 실행도 지원하며, GPU 실행에는 CUDA를 지원하는 PyTorch 설치가 필요합니다. `requirements.txt`는 버전 범위 명세이지 모든 환경을 고정한 lockfile은 아닙니다.

### 3. 질문 실행

```bash
python -m korean_rag --question "방탄소년단의 팬클럽 창단식은 몇 월 며칠인가?"
```

검증 환경의 출력 예시입니다.

```text
Answer: 2014년 3월 29일
Source: extractive_reader
Reader span score: 0.9591
Citation: 방탄소년단 (chunk 3167)
```

`--question`을 생략하면 질문 하나를 직접 입력할 수 있습니다. 대화 이력을 유지하는 챗봇은 아닙니다.

## 실행 옵션

| 용도 | 명령어 |
| --- | --- |
| 직접 질문 입력 | `python -m korean_rag` |
| 검색·리랭킹 결과만 확인 | `python -m korean_rag --no-reader --question "질문"` |
| CPU 실행 | `python -m korean_rag --device cpu --question "질문"` |
| CUDA GPU 실행 | `python -m korean_rag --device cuda --question "질문"` |
| JSON 출력 | `python -m korean_rag --json --question "질문"` |
| 다른 출력물 경로 사용 | `python -m korean_rag --outputs-root "D:/data/outputs" --question "질문"` |
| 전체 옵션 | `python -m korean_rag --help` |

`--no-reader`는 답을 추출하지 않고 검색 근거만 보여줍니다. 정확도를 높이는 옵션은 아닙니다. 검색 결과에 관련 없는 문맥이 섞일 수도 있습니다.

### 답변 보류 기준

후보마다 RRF 0.025, reranker 0.50, Reader 0.10 기준을 검사합니다. Reader의 빈 답 점수와 추출 점수를 비교하고, 답 문자열이 인용 문맥에 있는지도 확인합니다. 질문에 명시된 네 자리 연도(`2030년` 등)가 해당 근거에 없으면 보류합니다.

```bash
python -m korean_rag --question "내일 서울 날씨는 어때?"
```

검증 환경에서는 `Source: abstained`로 답변을 보류했습니다. 더 높은 점수의 후보만 허용하려면 다음처럼 실행합니다.

```bash
python -m korean_rag --question "질문" --min-rrf-score 0.03 --min-reranker-score 0.70 --min-reader-score 0.20
```

RRF는 순위 합산 점수이며 확률이 아닙니다. reranker나 Reader 점수도 정답 확률로 보정된 값이 아닙니다. 기준을 높이면 정답도 거절할 수 있습니다. JSON의 `answer_in_evidence` 역시 문자열 포함 검사일 뿐, 의미적 정답 판정은 아닙니다.

## 평가 결과

2026-09-05 실행 기준입니다. 검색은 1,000문항, 답변은 기존의 고정 100문항으로 평가했습니다.

| 항목 | 결과 |
| --- | ---: |
| Hybrid + Reranker Recall@5 | 0.9020 |
| Hybrid + Reranker Recall@20 | 0.9350 |
| 노트북 Reader-First 정답 포함 | 74/100 |
| 로컬 앱 정답 포함 | 67/100 |
| 로컬 앱 답변 반환 / 보류 | 79 / 21 |
| 반환한 답변 중 정답 포함 | 67/79 (84.8%) |
| 수작업 무응답 점검에서 보류 | 5/6 |

노트북과 로컬 앱은 답변 정책이 달라 결과가 같지 않습니다. 정답 포함은 정규화한 정답 문자열의 포함 여부이며 의미적 정확도와 다릅니다. 기존 개발 표본을 재사용했으므로 새 held-out 성능으로 해석하면 안 됩니다.

무응답 점검도 알려진 실패 사례를 포함한 6문항입니다. 특히 팬클럽 ‘해체식’이라는 잘못된 전제에 날짜를 반환하는 실패가 남아 있습니다. 이 프로젝트는 범용 챗봇이나 완벽한 오답 거절 시스템이 아닙니다.

[4주차 분석](reports/week4_experiment_results.md) · [실측 지표와 실행 환경](reports/local_evaluation/metrics.json) · [100개 답변](reports/local_evaluation/predictions.json) · [무응답 점검](reports/local_evaluation/unanswerable_probes.json)

## 직접 검증하기

코드와 저장된 결과의 정합성을 확인합니다. 모델 추론은 하지 않습니다.

```bash
python -m unittest discover -s tests -v
```

배포된 출력물이 있으면 현재 테스트 20개가 모두 실행됩니다. 출력물이 없으면 저장 결과를 읽는 테스트 3개는 건너뜁니다.

실제 검색부터 답변까지 100문항과 무응답 6문항을 다시 평가하려면:

```bash
python -m korean_rag.evaluate --device auto --save-dir transformer/outputs/local_verification
```

재학습하지 않으며, 결과 폴더에 `metrics.json`, `predictions.json`, `unanswerable_probes.json`을 저장합니다. 이 경로는 Git 업로드에서 제외됩니다. 같은 경로로 다시 실행하면 해당 검증 결과를 덮어쓰므로 이전 기록을 유지하려면 저장 경로를 바꾸세요.

## 코드와 실험 기록

| 경로 | 내용 |
| --- | --- |
| `korean_rag/` | 검색·리랭킹·답변 처리, CLI, 실제 추론 평가 |
| `transformer/notebooks/` | 1~4주차 학습·평가 노트북 |
| `reports/` | 주차별 분석과 로컬 검증 기록 |
| `tests/` | 회귀 테스트와 저장 결과 정합성 검사 |
| `scripts/refresh_week4_reports.py` | 저장된 답변의 인용·표·그림 재생성 |
| `requirements-reports.txt` | 보고서 재생성용 추가 의존성 |

노트북 실행 순서와 후처리 기록은 [노트북 안내](transformer/notebooks/README.md)에 정리했습니다. 로컬 앱을 사용하는 데 노트북 재학습이나 보고서 재생성은 필요하지 않습니다.
