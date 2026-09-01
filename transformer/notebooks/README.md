# 노트북 안내

이 폴더의 노트북은 한국어 RAG 시스템을 4주 동안 단계적으로 구축한 실험 기록입니다. 각 노트북에는 일차별 실험 과정과 핵심 결과가 남아 있습니다.

| 주차 | 노트북 | 핵심 질문 | 재사용 결과물 |
|---|---|---|---|
| 1주차 | `Week - 1/korean-transformer-finetuning.ipynb` | 한국어 Transformer를 어떻게 fine-tuning하고 평가하는가? | 학습 프로토콜과 분류 오류 분석 |
| 2주차 | `Week - 2/sentence-embedding-retrieval.ipynb` | Contrastive fine-tuning이 의미 기반 검색을 개선하는가? | Dense Retriever 체크포인트와 토크나이저 |
| 3주차 | `Week - 3/cross-encoder-reranker.ipynb` | Cross-Encoder가 검색 문맥의 순위를 개선하는가? | 최종 Reranker 체크포인트 |
| 4주차 | `Week - 4/hybrid-rag.ipynb` | Hybrid Retrieval과 근거 기반 QA로 사용할 수 있는 한국어 RAG를 만들 수 있는가? | BM25, FAISS, 지식베이스, RAG 평가 결과 |

## 아티팩트 연결

2주차 Retriever는 3주차와 4주차에서 사용합니다. 3주차 Reranker는 4주차에서 사용합니다. 로컬 터미널 애플리케이션은 최종적으로 `transformer/outputs/week2`, `week3`, `week4`의 아티팩트를 직접 불러옵니다.

## 데이터와 답변 범위

4주차의 최종 지식베이스는 **KorQuAD 1.0 한국어 위키백과 문맥**으로 구성했습니다. 따라서 노트북의 정량 평가는 KorQuAD 기반 검색·추출형 질의응답 성능을 나타냅니다.

이 결과를 일반 지식 챗봇의 성능으로 해석하면 안 됩니다. 최신 정보, 실시간 정보, 개인 정보, 지식베이스에 없는 주제는 답할 수 없으며, 로컬 애플리케이션은 근거가 부족할 때 답변을 거절하도록 구성했습니다.

## 노트북 재현 순서

1. 각 노트북을 새로운 Kaggle 세션에서 위에서 아래로 실행합니다.
2. 3주차 전에는 2주차 아티팩트를 Kaggle 입력으로 연결합니다.
3. 4주차 전에는 2주차 Retriever와 3주차 Reranker 아티팩트를 Kaggle 입력으로 연결합니다.
4. 포트폴리오에는 중간 학습 지표보다 최종 Retrieval과 RAG 평가 지표를 사용합니다.

## GitHub에서 로컬 실행하기

GitHub 용량 제한 때문에 100MB가 넘는 모델 체크포인트와 인덱스는 저장소에 포함하지 않습니다. 아래 Google Drive에서 `outputs` 폴더를 내려받아 `transformer/outputs` 위치에 넣어야 합니다.

[Google Drive - outputs 다운로드](https://drive.google.com/drive/folders/1wqPSshOLzAJHYGfR7Z-Ug0BP_k3ycL4s?usp=sharing)

그 다음 프로젝트 최상위 폴더에서 아래 명령어를 실행합니다.

```powershell
python -m pip install -r requirements.txt
python -m korean_rag --question "방탄소년단의 팬클럽 창단식은 몇 월 며칠인가?"
```

상세한 실행 옵션과 답변 거절 기준은 프로젝트 최상위 [README](../../README.md)를 참고하세요.
