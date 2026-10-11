# 공개 다운로드 검증

[English](public-download-verification.md) | [한국어](public-download-verification.ko.md)

2026-10-11에 실제 공개 소스 압축파일을 새로 내려받아 깨끗한 Windows CPU 환경에서 처음 실행하는 과정을 검증했습니다. 공개 `run.ps1 --doctor`가 새 격리 환경을 구성하고 세 모델 가중치와 메타데이터를 내려받았으며, 공식 KorQuAD 개발 원본 JSON으로 로컬 corpus와 chunk 파일을 재구성했습니다. 명세의 모든 파일 크기와 SHA-256이 일치했고 전체 모델 manifest 검사, `pip check`, doctor 검사도 통과했습니다. 비공개 모델이나 데이터 파일, 인증정보, 기존 Hugging Face 모델 캐시는 사용하지 않았습니다.

| 측정 항목 | 결과 |
| --- | --- |
| 공개 파일로 처음 준비하고 doctor 완료 | 223.43초 |
| 첫 persistent CLI 세션 | 질문 2개, 20.95초 |
| 준비된 환경의 `--no-download` 세션 | 질문 4개, 33.55초 |
| 전체 예시 추론 | 공개 문서의 서로 다른 예시 2개로 총 6회 |
| 모델 적재 | persistent 세션마다 한 번 |
| 반복 답변, 판단과 인용 식별자 | 반복 및 세션 사이 모두 일치 |
| Python / PyTorch | 3.12.7 / 2.14.0+cpu |
| 가상환경의 시스템 패키지 공유 | 없음 |

두 예시는 두 세션에서 모두 일관되게 `abstained`를 반환했습니다. 실행과 답변 보류 경로를 확인한 결과이며 답변 가능한 평가 질문의 정답률을 검증한 결과는 아닙니다. 실제 공개 실행 스크립트의 JSON 모드로 처리했고 `/quit`로 정상 종료했습니다. 재사용 세션에서는 `--no-download`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`과 로컬 파일만 사용하는 모델 적재를 적용했으며 제한된 셸 환경에서 실행했습니다. Hugging Face 캐시는 계속 비어 있었습니다. 별도의 네트워크 패킷 감시나 운영체제 방화벽 검증을 수행했다고 주장하지 않습니다.

이번 처음 실행에서는 기본 스크립트가 기존 Python 3.12를 사용했습니다. PATH에 Python이 없는 Windows x64와 Linux x64의 자동 설치는 앞선 별도 검증에서 실제 수행했습니다. 의존성은 `--no-cache-dir`로 새 가상환경에 내려받았습니다. 시간은 이번 한 번의 네트워크 상황과 컴퓨터 부하를 포함하므로 모든 사용자의 예상 설치 시간으로 해석하지 않습니다.

실제로 검증에 사용한 공개 소스 압축파일은 618,451바이트이며 SHA-256은 `88240025f30e35fb3e1a5928b4821f0defc7cb3e239120281cd71cfc260c8b02`입니다. 처음 배포된 검증 대상 압축파일을 식별하는 값으로, 이후 설명 문서만 추가한 압축파일의 해시는 달라질 수 있습니다. 동결된 핵심 소스 파일 19개의 내용은 설치와 두 실행 세션 뒤에도 동일했습니다. 파일별 식별자, 정확한 공개 URL과 패키지 버전은 [public-download-verification.json](public-download-verification.json)에 보존했습니다.

| 공개 파일 | 바이트 | SHA-256 |
| --- | ---: | --- |
| 실행 메타데이터 ZIP | 7,580,697 | `73537c36b5f33743b150db5fc7fcff91fb65cd2174693290fed75a3808f06d8f` |
| Reader 가중치, 고정한 공식 Hugging Face revision | 449,357,300 | `75ad3ca646cd32eefbc8f34cb17b34f191783edcf537fd2f3f865f25617240ce` |
| 최초 reranker 가중치, 기존 공개 Drive 파일 | 442,502,720 | `98071cd9882265c172865279b91f75e0b40c6f5e9787be60296ee82fb5f29d2c` |
| E5 retriever 가중치, 고정한 공식 Hugging Face revision | 470,641,600 | `1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477` |
| 변경하지 않은 공식 KorQuAD 개발 JSON | 3,881,058 | `25ffeb51e6c51ec02c071b60a10188e10005c144110f0d876b26079d80a35bdf` |

소스와 의존성을 제외한 실행 파일 다운로드 합계는 1,373,963,375바이트입니다. 메타데이터에는 원문 문단 대신 재구성에 필요한 위치와 식별자만 포함합니다. 설치 스크립트가 검증한 공식 원본에서 로컬 corpus와 chunk 파일을 생성한 뒤 전체 manifest를 검사합니다. 완성된 manifest의 SHA-256은 `2834c057f845137c129dcbbd6bc419e0a63303779d0a2f31a0b7cff11a4a57b9`이며 canonical 동결 식별자 `e2304cebdc5ed6af713f0a6a3376d15849394f882c718c81ce5ad4963d2b0574`와 연결됩니다.

같은 컴퓨터에서 새 소스와 공개 배포 경로를 확인한 검증이며 외부 제3자의 독립 검토는 아닙니다. Windows/Linux CPU 실행 검증과 synthetic 최종 CPU 감사는 [environment-verification.ko.md](environment-verification.ko.md), [synthetic-final-cpu-audit.ko.md](synthetic-final-cpu-audit.ko.md)에 각각 범위를 구분해 기록했습니다. 실행 방법은 [runtime.ko.md](../docs/runtime.ko.md)에서 확인할 수 있습니다.
