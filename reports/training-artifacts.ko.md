# 선택 사항인 학습 모델 압축 파일

[English](training-artifacts.md) | [한국어](training-artifacts.ko.md) | [크기, SHA-256, 파일 목록](training-artifacts.json)

선택 자료인 `korean-rag-training-models.zip`을 [Google Drive](https://drive.google.com/file/d/1attqOUViSx2i4q0JcRz2l577HZTGIuh6/view?usp=drivesdk)에서 공개합니다. 2026년 10월 11일에 계정 없이 [전체 파일 다운로드](https://drive.usercontent.google.com/download?id=1attqOUViSx2i4q0JcRz2l577HZTGIuh6&export=download&confirm=t)를 검증했습니다. 질의응답 실행에 필수인 파일은 아닙니다.

기존 프로젝트의 STS 검색 모델을 Hugging Face 형식으로 변환한 초기값, 추가 학습에서 선택한 1에포크 검색 모델, 경로와 개인 정보를 제외한 학습 및 입력 출처 기록, 제한된 예산 학습 설정, 영어와 한국어 안내, 원본 KLUE 출처 및 라이선스 고지를 포함합니다. Reader 가중치, 원본 KorQuAD 질문과 문단, 추출한 텍스트 쌍, 개인 기록은 제외했습니다. 압축 내부의 각 파일에 SHA-256을 기록했으며 전체 ZIP CRC와 파일별 해시를 확인했습니다.

## 공개 다운로드 검증

Google Drive 공유 설정은 **링크가 있는 모든 사용자, 뷰어**입니다. 전체 파일을 내려받은 응답은 HTTP 200과 `application/octet-stream`이었으며 크기는 정확히 **790,806,262바이트**, SHA-256은 `9cbb8be9921dcf96a15f95fcc5c55f075be1344ceb3a2b6836b8bf366ba72677`이었습니다. 인증정보, Authorization 헤더나 쿠키 없이 검증했습니다. 준비한 파일과 크기 및 해시가 같고 기록한 압축 내부 파일 목록도 유지합니다. 검증 시점과 다운로드 식별자는 [명세](training-artifacts.json)에 기록했습니다.

최종 질의응답 실행 구성은 multilingual E5와 별도의 실행 아티팩트 명세를 사용합니다. 이번 압축 파일은 기존 STS 검색 모델과 추가 학습 후보를 보존해 학습 과정을 확인하고 재현하는 자료입니다. 기존 Drive 폴더에 추가한 것이 최종 실행 모델을 교체하거나 공개 평가 수치를 변경하는 것은 아닙니다.

## 내려받은 파일 확인과 사용

```powershell
Get-FileHash -LiteralPath .\korean-rag-training-models.zip -Algorithm SHA256
(Get-Item -LiteralPath .\korean-rag-training-models.zip).Length
```

두 값을 [명세](training-artifacts.json)와 비교한 뒤 새 쓰기 가능 경로에 압축을 풀고 `README.md` 또는 `README.ko.md`를 따릅니다. `models/original-retriever`를 소스 저장소의 `.local/training-inputs/retriever`에 복사한 뒤 [학습 안내](../docs/training.ko.md)의 명령을 실행합니다. 원래 재순위화 모델은 별도 입력이며, 검증된 채택 실행 구성은 원래 가중치를 유지하므로 그 모델을 초기값으로 사용할 수 있습니다. 기록된 추가 학습을 재현할 때는 후보인 추가 학습 모델로 초기 모델을 대체하지 않습니다.

압축의 학습 기록은 내부 체크포인트 선택 결과이며 최종 RAG 정답률이 아닙니다. 초기 모델과 추가 학습 모델의 출처 해시 및 재순위화 비교의 해석 한계를 함께 제공합니다.
