# Synthetic 최종 질문의 추가 CPU 검증

[English](synthetic-final-cpu-audit.md) | [한국어](synthetic-final-cpu-audit.ko.md)

GPU 최종 평가가 끝난 뒤 동결된 최종 선택 모델을 별도의 깨끗한 Windows CPU 환경에서 synthetic 답변 불가 최종 질문 80개 전체로 추가 검증했습니다. 답변 가능한 최종 평가 질문의 CPU 추론은 수행하지 않았으며 이후 정책 조정이나 모델 선택도 하지 않았습니다.

| 측정 항목 | GPU 최종 평가 | 추가 CPU 검증 |
| --- | --- | --- |
| Synthetic 답변 불가 질문 | 80개 | 80개 |
| 잘못 응답한 질문 | 4개 | 4개 |
| 잘못 응답한 비율 | 5.00% | 5.00% |

CPU와 GPU의 응답 판단이 다른 질문은 0개였습니다. 소스, 설정, 모델 manifest, 동결 기록과 질문 파일 해시를 확인했습니다. 이 synthetic 질문들은 제한된 답변 불가 조건을 검사하며 모든 질문에서의 안전성이나 외부 제3자의 독립 검증을 뜻하지 않습니다. 답변 가능한 최종 질문 전체의 성능은 GPU 측정 결과입니다. 이번 환경의 CPU 모델 적재는 12.44초, synthetic 질문 80개 평균 추론 시간은 2.99초였습니다.

[synthetic-final-cpu-audit.json](synthetic-final-cpu-audit.json)에는 질문 식별자, 응답 판단, 해시와 실행 환경의 집계 측정만 공개합니다. 전체 예측은 개인 검증 폴더에 보존합니다. Windows와 Linux의 개발 질문 두 개 실행 비교는 별도의 [environment-verification.ko.md](environment-verification.ko.md)에서 확인할 수 있습니다.
