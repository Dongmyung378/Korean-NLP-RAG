# Week 4 결과 요약

## 목표

Week 2의 밀집 검색기와 Week 3 reranker를 연결해 한국어 Hybrid RAG 파이프라인을 만들었다. BM25, dense retrieval, RRF, reranking, 답변 추출, 인용을 한 흐름으로 평가했다.

## 검색 결과

| 방법 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.6090 | 0.7860 | 0.8240 | 0.8510 | 0.6870 | 0.6958 |
| Dense retrieval | 0.5490 | 0.7950 | 0.8510 | 0.9010 | 0.6535 | 0.6767 |
| Hybrid RRF | 0.6620 | 0.8590 | 0.9000 | 0.9350 | 0.7452 | 0.7604 |
| Hybrid RRF + reranker | 0.6660 | 0.9000 | 0.9210 | 0.9350 | 0.7661 | 0.7835 |

## 답변 전략 비교

| 방법 | 정답 포함률 | 근거 일치율 | 질의당 시간 |
| --- | ---: | ---: | ---: |
| Qwen Direct Top-3 | 0.55 | 0.74 | 0.422초 |
| Qwen Focused + Validation | 0.56 | 0.85 | 0.541초 |
| KorQuAD Extractive QA Reader | 0.64 | 0.85 | 0.031초 |
| Reader-First 최종 답변 | 0.64 | 0.85 | 0.031초 |

## 정리

BM25는 첫 순위에서 강했고, dense retrieval은 넓은 후보 범위에서 강했다. 두 결과를 RRF로 합치자 Recall@20은 0.9350까지 올라갔다. 그 뒤 Week 3 reranker를 적용하자 Recall@5, MRR@10, nDCG@10이 다시 개선됐다.

KorQuAD는 답이 문맥 안에 있는 추출형 데이터라서, 생성형 Qwen만 쓰는 것보다 extractive QA Reader가 더 잘 맞았다. 최종 Reader-First 방식은 고정된 100개 질의에서 정답 포함률 0.64를 기록했고, 85%의 답변에 근거 chunk 인용을 남겼다.

이 시스템은 일반 지식 전체를 답하는 챗봇이 아니다. KorQuAD 지식베이스에서 근거를 찾을 수 있는 질문에 답하며, 근거가 부족하면 답변을 보류하는 방식이다.

## 산출물

- chunked KorQuAD knowledge base와 FAISS 인덱스
- BM25, dense, hybrid, reranker 비교 결과
- Qwen과 extractive Reader 답변 예시
- 답변 인용, 오류 분석, latency와 모델 효율 파일
- `week4_final_artifacts.zip`
