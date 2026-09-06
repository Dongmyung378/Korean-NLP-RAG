# Week 3 결과 요약

## 목표

Week 2의 밀집 검색 결과를 입력으로 사용해, 질문과 문맥을 함께 읽는 Cross-Encoder reranker를 학습했다. 평가는 분류 점수와 검색 순위 지표를 분리해 기록했다.

## 실험 결과

| 구분 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Week 2 dense retrieval | 0.4610 | 0.6460 | 0.7380 | 0.8180 | 0.5437 | 0.5899 |
| Dense retrieval + reranker | 0.7520 | 0.8100 | 0.8160 | 0.8180 | 0.7796 | 0.7887 |

learning-rate 비교에서는 3e-5가 가장 높은 validation Macro-F1 0.9732를 기록했다. 이후 저장한 최종 체크포인트는 learning rate 1e-5, weight decay 0.01으로 재학습됐고 validation Macro-F1 0.9750을 기록했다. Week 4는 이 저장된 체크포인트를 사용했다.

## 정리

### 기준 모델 비교의 한계

3일차 random-negative 기준 모델은 hard-negative 대상 질문들을 포함해 학습했다. 이후 4일차에 질문 ID를 train/validation으로 나눈 뒤 같은 기준 모델을 validation에 평가했으므로, 검증 질문 일부를 이미 학습한 상태다. 따라서 `hard_negative_baseline_macro_f1`과 4일차 모델의 차이를 독립적인 검증이나 hard-negative 학습 효과의 확정적 증거로 해석하지 않는다.

최종 모델은 사전학습 가중치를 다시 불러와 4일차 train 분할로 학습했고, 위 검색 순위 표는 별도의 KorQuAD validation 질의에서 측정했다. 기준 모델의 질문 중복 문제와 이 순위 평가는 구분한다. 동일한 질문 분할에서 random-negative와 hard-negative 모델을 각각 학습한 통제 실험은 수행하지 않았다.

reranker는 Top-20 후보 집합을 늘리지는 않지만, 그 안의 순서를 크게 개선했다. Recall@20이 같다는 점은 후보 생성은 Week 2 retriever가 담당하고, Week 3 모델은 상위 순위 품질을 높였다는 뜻이다.

가장 큰 변화는 Recall@1과 MRR@10이다. 이 결과는 Week 4에서 hybrid retrieval의 Top-20을 다시 정렬하는 구조를 선택한 근거가 됐다.

## 산출물

- 하드 네거티브 학습 데이터
- 최종 Cross-Encoder 체크포인트
- dense rank와 reranker rank가 함께 저장된 후보 파일
- ranking evaluation 결과와 오류 사례
