# 외부 자료 고지

[English](THIRD_PARTY_NOTICES.md) | [한국어](THIRD_PARTY_NOTICES.ko.md)

## 데이터

KorQuAD 1.0의 제작자와 원문 기여자의 작업을 사용합니다. [공식 데이터 안내](https://korquad.github.io/category/1.0_KOR.html)는 [CC BY-ND 2.0 KR](https://creativecommons.org/licenses/by-nd/2.0/kr/deed.ko)을 명시합니다. 공개 압축 파일에는 변형한 질의응답 원문 대신 문항 ID, 해시와 재구성 방법을 제공합니다. 변경하지 않은 공식 JSON을 내려받아 사용자 컴퓨터에서 실행용 원문과 평가 파일을 생성합니다.

## 모델

| 구성 요소 | 원본 | 출처와 사용 범위 |
| --- | --- | --- |
| E5 검색기 | [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3) | 고정한 모델 카드는 MIT를 명시합니다. E5 저자와 기여자의 모델이며 프로젝트에서는 접두어와 로컬 문서 인덱스를 적용합니다. |
| 프로젝트 검색기와 재순위화 모델 | [klue/roberta-base](https://huggingface.co/klue/roberta-base/tree/02f94ba5e3fcb7e2a58a390b8639b0fac974a8da) | KLUE-RoBERTa에서 STS, 관련성 분류와 질문 및 문단 검색을 추가 학습했습니다. Sungjoon Park 외 KLUE-benchmark 저자와 기여자의 작업에 기반합니다. |
| QA Reader | [monologg/koelectra-base-v3-finetuned-korquad](https://huggingface.co/monologg/koelectra-base-v3-finetuned-korquad/tree/ad4097ba1dc9904a22c3df71cafe6431e23f8914) | monologg와 KoELECTRA 기여자의 모델이며 변경하지 않은 가중치를 원본 저장소에서 내려받습니다. |

[KLUE 원본 저장소](https://github.com/KLUE-benchmark/KLUE/tree/3efd98708a40ff49251fddde35453f8fbb11f536)는 CC BY-SA 4.0을 명시하며 [원본 고지](licenses/klue-cc-by-sa-4.0.md)를 보존했습니다. 고정한 Hugging Face 체크포인트에는 별도의 라이선스 태그가 없습니다. 모델 파생물에 원본 선언을 적용하는 것은 원본 작업을 근거로 한 해석이며 프로젝트의 추가 학습 내용은 결과 기록에서 구분합니다.

[KoELECTRA 원본 저장소](https://github.com/monologg/KoELECTRA/tree/024fbdd600e653b6e4bdfc64ceec84181b5ce6c4)와 기반 discriminator는 Apache-2.0을 명시하며 [원본 고지](licenses/koelectra-apache-2.0.txt)를 보존했습니다. 실제 사용하는 Reader의 고정 버전에는 모델 카드, 라이선스 태그와 파일이 없습니다. 해당 체크포인트에 별도로 확인한 라이선스를 부여하거나 가중치를 재배포하지 않습니다.

참고 자료: [KLUE 논문](https://arxiv.org/abs/2105.09680), [E5 논문](https://arxiv.org/abs/2212.03533), [KoELECTRA 프로젝트](https://github.com/monologg/KoELECTRA). 패키지에는 각각의 원본 라이선스가 적용됩니다. 이 고지는 프로젝트 자체 소스 코드에 새 라이선스를 부여하지 않습니다.
