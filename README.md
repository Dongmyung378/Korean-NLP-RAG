# Korean NLP RAG

한국어 Transformer 학습부터 Dense Retrieval, Cross-Encoder Reranking, Hybrid Retrieval, 근거 기반 질의응답까지 단계적으로 구현한 포트폴리오 프로젝트입니다.

## 프로젝트 목표

- 한국어 Transformer fine-tuning과 정량 평가
- 문장 임베딩 기반 Dense Retrieval 구축
- Hard-negative Cross-Encoder Reranker 학습
- BM25와 Dense Retrieval을 결합한 RRF Hybrid Retrieval 구현
- 검색 근거를 인용하는 한국어 RAG 터미널 애플리케이션 구현
- Retrieval과 답변 품질에 대한 오류 분석

## 최종 파이프라인

`질문 -> BM25 Top-50 + Dense Retrieval Top-50 -> RRF Hybrid Top-20 -> Cross-Encoder Reranking Top-3 -> KorQuAD QA Reader -> 답변과 근거 인용`

## 데이터 범위와 사용 시 유의 사항

이 프로젝트의 최종 지식베이스는 **KorQuAD 1.0의 한국어 위키백과 문맥**을 청크 단위로 분할해 만들었습니다. 따라서 역사, 인물, 문화, 과학 등 KorQuAD 문맥에 포함된 주제에는 답할 수 있지만, 모든 일반 질문에 답하는 챗봇은 아닙니다.

- 최신 뉴스, 실시간 날씨, 개인 정보, 전문 상담처럼 지식베이스에 없는 정보는 답할 수 없습니다.
- 프로그래밍, 생활 정보 등도 관련 근거가 저장되어 있지 않으면 답변을 거절할 수 있습니다.
- 답변은 검색된 근거 문맥에서 추출하며, 검색·리랭킹·Reader 신뢰도가 낮거나 답변이 인용 근거에 없으면 답을 추측하지 않습니다.
- 따라서 결과에 표시되는 인용 문맥을 함께 확인하는 것이 좋습니다.

## 실행 전: 출력물 다운로드

학습된 Week 2 Retriever와 Week 3 Reranker는 각각 100MB를 넘기 때문에 GitHub 저장소에 포함하지 않습니다. 아래 Google Drive에서 `outputs` 폴더를 내려받은 뒤, 프로젝트의 `transformer` 폴더 안에 배치해야 합니다.

[Google Drive - outputs 다운로드](https://drive.google.com/drive/folders/1wqPSshOLzAJHYGfR7Z-Ug0BP_k3ycL4s?usp=sharing)

다운로드 후 폴더 구조는 아래와 같아야 합니다.

```text
Korean-NLP-RAG/
├─ korean_rag/
├─ transformer/
│  ├─ notebooks/
│  └─ outputs/
│     ├─ week2/
│     ├─ week3/
│     └─ week4/
└─ requirements.txt
```

`outputs` 폴더가 없다면 터미널 애플리케이션은 실행되지 않습니다.

## 터미널 실행 방법

프로젝트 최상위 폴더에서 실행합니다.

```powershell
cd C:\Users\leona\Desktop\Korean-NLP-RAG
python -m pip install -r requirements.txt
python -m korean_rag --question "방탄소년단의 팬클럽 창단식은 몇 월 며칠인가?"
```

일반 실행은 BM25, Dense Retrieval, RRF, Cross-Encoder, KorQuAD QA Reader를 차례로 수행하고 답변과 인용 근거를 출력합니다. 처음 한 번만 공개 KorQuAD QA Reader 모델을 다운로드하며, 이후에는 로컬 캐시를 사용합니다.

### 자주 쓰는 명령어

| 용도 | 명령어 |
|---|---|
| 질문을 직접 입력 | `python -m korean_rag` |
| 검색·리랭킹 근거만 확인 | `python -m korean_rag --no-reader --question "질문"` |
| CPU 사용 | `python -m korean_rag --device cpu --question "질문"` |
| CUDA GPU 사용 | `python -m korean_rag --device cuda --question "질문"` |
| JSON 형식 출력 | `python -m korean_rag --json --question "질문"` |
| 전체 옵션 확인 | `python -m korean_rag --help` |

`--no-reader`는 최종 답을 만들지 않고, 검색과 리랭킹으로 찾은 근거 문맥만 보여주는 검증용 옵션입니다.

### 답변 거절 기준

기본 답변 모드는 Hybrid Retrieval, Cross-Encoder, QA Reader 신뢰도를 확인합니다. 더 보수적으로 답변을 거절하고 싶다면 아래처럼 기준을 높일 수 있습니다.

```powershell
python -m korean_rag --question "질문" --min-rrf-score 0.03 --min-reranker-score 0.70 --min-reader-score 0.20
```

기준을 높이면 근거 없는 답변 위험은 줄지만, 답변 거절 횟수는 늘어납니다.

## 노트북과 실험 결과

주차별 실험 실행 순서와 재사용 아티팩트는 [노트북 안내](transformer/notebooks/README.md)에서 확인할 수 있습니다.
