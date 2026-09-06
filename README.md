# Korean-NLP-RAG

한국어 문맥에서 질문에 필요한 근거를 찾고, 답과 출처를 터미널에 보여주는 프로젝트입니다. 4주 동안 Transformer 학습, 문장 임베딩, 검색, 리랭킹을 실험한 뒤 저장된 모델을 로컬 Python 코드로 연결했습니다.

최종 앱은 인터넷을 검색하거나 자유롭게 문장을 생성하지 않습니다. KorQuAD 문맥을 검색한 다음 QA Reader가 답을 추출합니다. Qwen을 이용한 생성형 RAG와의 비교 실험은 4주차 노트북에 남겨두었습니다.

## 구성

| 단계 | 작업 | 다음 단계에서 쓰는 결과물 |
| --- | --- | --- |
| 1주차 | KLUE-YNAT 뉴스 제목 분류, 학습 설정 비교 | 학습·평가 경험. 분류 모델은 검색기에 연결하지 않음 |
| 2주차 | KLUE-STS 문장 임베딩, hard-negative 추가 학습, FAISS 검색 | Retriever와 토크나이저 |
| 3주차 | Cross-Encoder 학습, 검색 후보 재정렬 | Reranker |
| 4주차 | BM25·Dense 검색 결합, Qwen·Reader 비교, 인용과 오류 분석 | 지식베이스, 인덱스, 평가 결과 |

로컬 앱의 처리 순서는 다음과 같습니다.

`질문 -> BM25 Top-50 + Dense Top-50 -> RRF Top-20 -> Reranker Top-3 -> QA Reader -> 답변과 인용 또는 답변 보류`

## 답변할 수 있는 범위

지식베이스는 KorQuAD 1.0 validation split의 한국어 위키백과 문맥 960개를 나눈 5,889개 청크입니다. 저장된 인물·문화·역사·과학 등의 문맥에서 사실을 찾을 수 있지만, 최신 뉴스나 날씨, 개인정보처럼 여기에 없는 정보에는 신뢰할 만한 답을 제공할 수 없습니다. 웹 검색, 사용자 문서 업로드, 대화 이력 기능은 없습니다.

관련 문맥을 찾았더라도 질문의 전제를 잘못 이해하거나 엉뚱한 구절을 추출할 수 있습니다. 인용이 붙어 있다는 이유만으로 정답이라고 판단하지 말고, 함께 출력되는 근거를 확인해 주세요.

## 실행하기

Python 3.12 환경에서 프로젝트 루트를 기준으로 실행합니다. 아래 명령은 Git Bash에서도 사용할 수 있습니다.

### 저장소와 출력물 준비

```bash
git clone https://github.com/Dongmyung378/Korean-NLP-RAG.git
cd Korean-NLP-RAG
```

이미 저장소가 있다면 그 폴더로 이동하면 됩니다.

모델 체크포인트는 각각 100MB가 넘어 GitHub에서 제외했습니다. [Google Drive에서 outputs 폴더](https://drive.google.com/drive/folders/1wqPSshOLzAJHYGfR7Z-Ug0BP_k3ycL4s?usp=sharing)를 받아 `transformer` 안에 넣어 주세요. ZIP으로 받았다면 압축을 풀고 폴더 구조를 유지해야 합니다.

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

주요 파일만 표시한 구조입니다. 개별 파일을 골라 받기보다 배포된 폴더를 그대로 옮기는 편이 편합니다. `outputs/outputs/`처럼 폴더가 중복되지 않도록 확인하세요. 1주차 출력물은 로컬 앱 실행에 필요하지 않습니다.

체크포인트와 BM25 파일은 역직렬화해서 읽습니다. 출처를 신뢰할 수 없는 파일은 사용하지 마세요.

### 설치와 첫 질문

```bash
python -m pip install -r requirements.txt
python -m korean_rag --question "방탄소년단의 팬클럽 창단식은 몇 월 며칠인가?"
```

검증 환경에서는 다음 답변과 인용이 나왔습니다.

```text
Answer: 2014년 3월 29일
Source: extractive_reader
Reader span score: 0.9591
Citation: 방탄소년단 (chunk 3167)
```

첫 실행에는 공개 QA 모델 `monologg/koelectra-base-v3-finetuned-korquad`를 내려받기 위한 인터넷 연결이 필요합니다. 이후에는 로컬 캐시를 사용합니다. 질문 한 번을 실행할 때마다 모델을 메모리에 다시 불러옵니다.

Windows, Python 3.12.7, PyTorch 2.5.1+cu121, Transformers 5.16.1에서 검증했습니다. CPU로도 실행할 수 있고, GPU를 쓰려면 CUDA 지원 PyTorch가 설치돼 있어야 합니다. 의존성은 버전 범위로 지정했으므로 환경이 달라지면 아래 테스트로 확인해 주세요.

## 자주 쓰는 명령

| 용도 | 명령어 |
| --- | --- |
| 질문을 직접 입력 | `python -m korean_rag` |
| 검색 결과만 확인 | `python -m korean_rag --no-reader --question "질문"` |
| CPU 사용 | `python -m korean_rag --device cpu --question "질문"` |
| GPU 사용 | `python -m korean_rag --device cuda --question "질문"` |
| JSON 출력 | `python -m korean_rag --json --question "질문"` |
| 다른 출력물 경로 사용 | `python -m korean_rag --outputs-root "D:/data/outputs" --question "질문"` |
| 전체 옵션 | `python -m korean_rag --help` |

`--no-reader`는 답을 추출하지 않고 검색·리랭킹 근거만 보여줍니다. 정확도를 높이는 옵션은 아닙니다.

질문은 앞뒤 공백을 제외하고 1,000자, 모델 토크나이저 기준 128토큰 이내로 받습니다. 둘 중 하나라도 넘으면 줄여 달라는 메시지를 출력합니다. 질문을 임의로 잘라 의미가 바뀌는 것을 피하기 위한 제한입니다.

### 답을 보류하는 경우

```bash
python -m korean_rag --question "내일 서울 날씨는 어때?"
```

이 질문은 검증에서 `Source: abstained`로 보류했습니다. 후보별 기본 기준은 RRF 0.025, reranker 0.50, Reader 0.10입니다. Reader가 빈 답을 더 높게 평가하는 경우와 인용 문맥에 답 문자열이 없는 경우도 제외합니다. 질문에 명시된 네 자리 연도(`2030년` 등)가 근거에 없을 때도 보류합니다.

기준은 실행 옵션으로 바꿀 수 있습니다.

```bash
python -m korean_rag --question "질문" --min-rrf-score 0.03 --min-reranker-score 0.70 --min-reader-score 0.20
```

이 점수들은 정답 확률이 아닙니다. 기준을 높이면 정답도 거절할 수 있고, 높은 점수의 오답이 남을 수도 있습니다. JSON의 `answer_in_evidence`는 답 문자열이 문맥에 있다는 뜻이지 정답 판정이 아닙니다.

## 결과와 남은 한계

2026-09-05의 저장 결과입니다. 검색은 1,000문항, 답변은 기존의 고정 100문항으로 평가했습니다.

| 항목 | 결과 |
| --- | ---: |
| Hybrid + Reranker Recall@5 | 0.9020 |
| Hybrid + Reranker Recall@20 | 0.9350 |
| 노트북 Reader-First 정답 포함 | 74/100 |
| 로컬 앱 정답 포함 | 67/100 |
| 로컬 앱 답변 반환 / 보류 | 79 / 21 |
| 반환한 답변 중 정답 포함 | 67/79 (84.8%) |
| 수작업 무응답 점검에서 보류 | 5/6 |

노트북은 로컬 앱과 같은 후보별 임계값을 사용하지 않아 수치가 다릅니다. 정답 포함은 정규화한 정답 문자열이 답변에 있는지를 계산한 값입니다. 의미적 정확도나 새 held-out 성능은 아닙니다.

무응답 점검도 알려진 실패 사례를 포함한 6문항에 한정됩니다. 팬클럽 ‘해체식’이라는 잘못된 전제의 질문에 날짜를 답하는 실패가 남아 있습니다.

3주차의 random-negative 기준 모델 비교에는 검증 질문 중복이 있어 hard-negative 학습 효과를 입증하는 근거로 사용할 수 없습니다. 독립된 질문 분할로 두 모델을 다시 비교한 결과는 없습니다. 최종 모델의 검색 순위 평가와의 차이는 [3주차 보고서](reports/week3_experiment_results.md)에 설명했습니다.

[4주차 분석](reports/week4_experiment_results.md), [실측 지표와 실행 환경](reports/local_evaluation/metrics.json), [100개 답변](reports/local_evaluation/predictions.json), [무응답 점검 결과](reports/local_evaluation/unanswerable_probes.json)에서 세부 내용을 볼 수 있습니다. 평가 기록의 코드 해시는 입력 검사와 예외 처리가 추가되기 전의 실행 버전을 가리킵니다.

## 검증 명령

코드와 저장 결과를 확인합니다. 모델을 학습하거나 추론하지 않습니다.

```bash
python -m unittest discover -s tests -v
```

출력물이 없으면 해당 파일을 읽는 테스트는 건너뜁니다. 실제 검색과 답변을 다시 평가하려면 다음 명령을 사용합니다.

```bash
python -m korean_rag.evaluate --device auto --save-dir transformer/outputs/local_verification
```

100개 질문과 무응답 6문항을 실행하고 `metrics.json`, `predictions.json`, `unanswerable_probes.json`을 저장합니다. 재학습은 하지 않습니다. 같은 경로를 다시 사용하면 기존 검증 파일을 덮어쓰므로 보관하려면 저장 경로를 바꿔 주세요. 이 결과 폴더는 Git 업로드에서 제외됩니다.

## 파일 안내

- `korean_rag/`: 검색, 리랭킹, 답변 처리, CLI와 평가 코드
- `transformer/notebooks/`: 1~4주차 실험 노트북
- `reports/`: 주차별 분석과 검증 기록
- `tests/`: 회귀 테스트와 저장 결과 검사
- `scripts/refresh_week4_reports.py`: 저장된 답변의 인용·표·그림 재생성

주차별 실행 순서와 보고서 재생성 방법은 [노트북 안내](transformer/notebooks/README.md)에 있습니다. 로컬 앱을 쓰는 데 노트북 재학습은 필요하지 않습니다.
