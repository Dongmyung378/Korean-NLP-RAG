# 선택 사항인 학습 모델 압축 파일

[English](training-artifacts.md) | [한국어](training-artifacts.ko.md) | [크기, SHA-256, 파일 목록](training-artifacts.json)

`korean-rag-training-models.zip`을 수동 업로드할 수 있도록 준비했습니다. 공개 다운로드 링크는 아직 없으며 주소를 임의로 기재하지 않았습니다. 이 압축 파일은 선택 자료이고 질의응답 실행에 필수인 파일은 아닙니다.

기존 프로젝트의 STS 검색 모델을 Hugging Face 형식으로 변환한 초기값, 추가 학습에서 선택한 1에포크 검색 모델, 경로와 개인 정보를 제외한 학습 및 입력 출처 기록, 제한된 예산 학습 설정, 영어와 한국어 안내, 원본 KLUE 출처 및 라이선스 고지를 포함합니다. Reader 가중치, 원본 KorQuAD 질문과 문단, 추출한 텍스트 쌍, 개인 기록은 제외했습니다. 압축 내부의 각 파일에 SHA-256을 기록했으며 전체 ZIP CRC와 파일별 해시를 확인했습니다.

## 준비한 파일 공개

1. 준비된 `korean-rag-training-models.zip`을 변경 없이 Google Drive에 업로드합니다. 파일명과 JSON 명세의 정확한 크기 및 해시를 유지합니다.
2. 공유 설정의 일반 액세스를 **링크가 있는 모든 사용자**, 권한을 **뷰어**로 설정하고 실제 공유 링크를 복사합니다.
3. 로그아웃한 브라우저나 시크릿 창에서 링크를 확인합니다. 완전한 ZIP을 내려받아 크기와 SHA-256이 명세와 같은지 비교합니다. 미리보기만 열리거나 별도 계정 승인이 필요한 상태는 공개 다운로드 검증이 완료된 상태가 아닙니다.
4. 비로그인 다운로드와 해시 확인을 마친 실제 링크만 이 안내와 `training-artifacts.json`에 기재합니다. 실제 업로드를 검증하기 전에는 `download_url`과 `drive_file_id`를 `null`로 유지합니다.

코딩 에이전트는 이 파일을 업로드하지 않았습니다. 소스 저장소의 실행기는 별도 실행 아티팩트 명세를 사용하므로 선택 학습 파일의 업로드가 채택된 실행 구성을 변경하지 않습니다.

## 내려받은 파일 확인과 사용

```powershell
Get-FileHash -LiteralPath .\korean-rag-training-models.zip -Algorithm SHA256
(Get-Item -LiteralPath .\korean-rag-training-models.zip).Length
```

두 값을 [명세](training-artifacts.json)와 비교한 뒤 새 쓰기 가능 경로에 압축을 풀고 `README.md` 또는 `README.ko.md`를 따릅니다. `models/original-retriever`를 소스 저장소의 `.local/training-inputs/retriever`에 복사한 뒤 [학습 안내](../docs/training.ko.md)의 명령을 실행합니다. 원래 재순위화 모델은 별도 입력이며, 검증된 채택 실행 구성은 원래 가중치를 유지하므로 그 모델을 초기값으로 사용할 수 있습니다. 기록된 추가 학습을 재현할 때는 후보인 추가 학습 모델로 초기 모델을 대체하지 않습니다.

압축의 학습 기록은 내부 체크포인트 선택 결과이며 최종 RAG 정답률이 아닙니다. 초기 모델과 추가 학습 모델의 출처 해시 및 재순위화 비교의 해석 한계를 함께 제공합니다.
