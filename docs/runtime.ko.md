# 실행 환경과 재현

[English](runtime.md) | [한국어](runtime.ko.md)

공개 실행 스크립트는 프로젝트 내부에 Python 환경을 준비하고, 기록된 모델 파일을 검증한 뒤 한국어 질의응답 세션을 시작합니다. 기본 실행 장치는 CPU입니다. Docker, 외부 LLM, API 키 또는 전역 패키지 설치가 필요하지 않습니다.

## 소스에서 실행하기

저장소를 복제하거나 [포트폴리오 소스 압축 파일](https://park-dongmyung.com/downloads/korean-nlp-rag-source.zip)을 쓰기 가능한 폴더에 압축 해제합니다. 내부 폴더 구조를 유지하고 해당 폴더에서 실행합니다.

```powershell
# Windows PowerShell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

Windows에서는 `launch-rag.bat`을 더블 클릭해도 됩니다. Linux에서는 다음 명령을 사용합니다.

```sh
sh run.sh
```

최초 실행에는 필요한 Python, 고정된 버전의 의존성, 모델 메타데이터와 가중치를 내려받기 위한 인터넷 연결이 필요합니다. 사용 가능한 64비트 Python 3.12가 있으면 이를 사용합니다. 없으면 지원 대상인 Windows x64와 Linux x64에서 공식 uv의 고정된 배포 파일을 검증하고 Python 3.12.15를 `.local/` 아래에 설치합니다. Python 다운로드, uv 캐시, `.venv`는 프로젝트 내부에 저장됩니다. Linux에서 이 자동 준비가 필요한 경우에만 `curl` 또는 `wget`, `tar`, `sha256sum`이 필요합니다.

질문을 한 줄씩 입력합니다. `/quit` 또는 입력 종료로 세션을 끝냅니다. 모델은 한 번 적재한 뒤 재사용합니다. 고정된 문서 집합에 답이 없는 질문은 답변을 보류할 수 있습니다.

## 단일 질문, 일괄 실행, CUDA

```powershell
.\run.ps1 --question "직지심체요절은 언제 인쇄되었나요?" --json
.\run.ps1 --questions-file .\questions.txt --json
.\run.ps1 --no-download --interactive
```

`questions.txt`는 한 줄에 질문 하나를 넣은 UTF-8 파일이어야 합니다. 일괄 실행 결과는 질문마다 JSON 객체 하나를 출력합니다. Linux에서도 `sh run.sh` 뒤에 같은 인자를 지정합니다.

CUDA는 선택 사항이며 명시적으로 요청해야 합니다.

```powershell
.\run.ps1 --device cuda --question "직지심체요절은 언제 인쇄되었나요?" --json
```

이 명령은 CUDA 13.0용으로 고정된 PyTorch 배포를 준비하고 호환되는 NVIDIA 장치와 드라이버가 있는지 확인합니다. 실행 환경을 변경할 때 기존 프로젝트 환경이 호환되지 않으면 `.local/environment-backups/`로 보존한 뒤 새 환경을 만듭니다. 실행 스크립트는 GPU 드라이버를 설치하거나 변경하지 않습니다. GPU 성능 수치는 함께 기록된 하드웨어와 실행 환경을 기준으로 해석해야 합니다.

## 환경 준비와 진단

```powershell
# 모델을 내려받거나 적재하지 않고 의존성만 준비
.\run.ps1 --setup-only

# 모델을 준비하고 설치 버전과 모든 파일을 검증
.\run.ps1 --doctor

# 다운로드 없이 기존 모델 파일만 검증
.venv\Scripts\python.exe scripts\download_artifacts.py --destination artifacts\runtime --verify-only

# 준비된 환경에서 CLI 직접 실행
.venv\Scripts\python.exe -m korean_rag --interactive --bundle artifacts\runtime --device cpu
```

Linux에서는 Python 경로를 `.venv/bin/python`으로 바꿉니다. 환경 준비만 직접 실행하려면 `python scripts/setup_runtime.py --device cpu`를 사용합니다. `--repair`는 호환되지 않는 프로젝트 환경을 보존한 뒤 교체합니다. 모델 준비 명령은 `python scripts/download_artifacts.py --manifest reports/artifact-download.json --destination artifacts/runtime`입니다. 두 스크립트는 모듈 실행도 지원하며, 진단 예시는 `python -m scripts.doctor --bundle artifacts/runtime`입니다.

`--doctor`는 Python과 고정된 패키지 버전, 모델 파일 검증 결과, 문제를 JSON으로 출력합니다. 파일을 확인하는 기능이며 답변 품질을 측정하지 않습니다. 누락되거나 변경된 파일은 모델 적재 전에 거부합니다. 불완전한 모델 폴더를 정상으로 간주하거나 덮어쓰지 않으므로 원인을 확인한 뒤 새로운 `--bundle` 경로를 선택합니다. 정상적으로 준비된 환경과 모델은 `--no-download`로 오프라인에서 재사용할 수 있습니다.

## 모델 출처와 원문 데이터의 로컬 재구성

`reports/artifact-download.json`에는 공개 다운로드 URL, 정확한 바이트 수, SHA-256 해시가 기록됩니다. 작은 메타데이터 ZIP은 포트폴리오에서 제공하고, 변경하지 않은 공개 모델 가중치는 기록된 Hugging Face 공식 revision에서 내려받습니다. 프로젝트에서 학습한 가중치는 기록된 공개 파일 링크를 사용합니다. GitHub 인증에 의존하지 않고 원래 모델의 가중치를 다시 배포하지 않도록 구성한 방식입니다. 압축 파일을 먼저 검증한 뒤 해제하며 별도로 받는 모델 파일도 검증을 통과해야 전체 모델 묶음을 사용할 수 있습니다. 상위 경로 접근, 파일 시스템 링크, 허용하지 않는 파일 형식, 중복 파일명을 거부하고 개별 파일이 압축 파일의 기존 항목을 대체하지 못하게 합니다. 최종 manifest는 모델, 토크나이저, 인덱스, 설정, 원문 파일의 전체 목록을 검증합니다.

메타데이터 압축 파일에는 모델 가중치, `corpus.jsonl`, `chunks.jsonl`을 포함하지 않습니다. 다운로드 명세의 `data_recipe`는 변경되지 않은 [공식 KorQuAD 검증 원본](https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json)을 내려받아 바이트 수와 해시를 확인하고, 문단 식별자와 문자 위치로 원문과 청크를 사용자 환경에서 재구성합니다. 생성한 파일의 정확한 바이트도 모델 manifest와 대조합니다. 공개 재구성 명세에는 복사한 문단이나 질문 대신 식별자와 위치 정보가 들어갑니다. KorQuAD의 이용 조건은 CC BY-ND 2.0 KR이며 저장소의 데이터 출처와 라이선스 안내를 함께 확인합니다.

실행 시에는 로컬 Hugging Face `safetensors` 모델을 사용하며 별도의 모델 코드나 pickle 체크포인트를 내려받지 않습니다. BM25는 검증한 청크에서 다시 구성합니다. 소스 파일의 바이트는 Windows와 Linux에서 유지되지만, 동일한 파일이 모든 CPU와 GPU에서 비트 단위로 동일한 부동소수점 결과를 보장하지는 않습니다.

## 평가 데이터와 성능 재현

환경 준비 후 기록된 공식 원본에서 실제 평가 분할을 재구성합니다.

```powershell
.venv\Scripts\python.exe scripts\reproduce_data.py --source-dir .local\source-data --output-dir .local\benchmark-v1
```

출력 폴더는 아직 존재하지 않는 경로여야 합니다. 재구성 명세는 분할 간 분리, 측정에 사용한 데이터 수, 파일 해시를 확인합니다. 검증된 원본 파일이 이미 원본 폴더에 있다면 `--offline`을 사용할 수 있습니다.

개발용 데이터 평가는 다음과 같습니다.

```powershell
.venv\Scripts\python.exe scripts\evaluate_portable.py --bundle artifacts\runtime --benchmark .local\benchmark-v1 --split dev --device cpu --output .local\evaluation-dev
```

평가 출력 폴더도 새로운 경로여야 합니다. 예측값, 지표, 소스와 모델 해시, 실행 환경, 적재 시간과 추론 시간을 기록합니다. CPU에서 전체 평가는 오래 걸릴 수 있습니다. 최종 평가에는 `--freeze`로 공개된 동결 기록을 추가해야 하며, 최종 질문을 읽기 전에 소스, 설정, 입력 해시가 기록과 일치해야 합니다. 답변 불가 평가를 함께 수행한 경우처럼 정확한 최종 입력 구성은 평가 보고서를 따릅니다. 임의로 만든 질문 목록으로는 공개된 평가 점수를 재현할 수 없습니다.

### 최초 개발 retriever의 명시적 준비

선택된 실행 모델은 `.pt` 체크포인트를 사용하지 않습니다. 최초 개발 모델과의 비교를 재현하려면 [원래 프로젝트 체크포인트](https://drive.google.com/file/d/1kairfruRDX3xTYU3u_8h4VdZwoMyE887/view)를 직접 내려받아 `.local/original/week2_dense_retriever.pt`에 저장합니다. [최초 모델 메타데이터 압축 파일](https://park-dongmyung.com/downloads/korean-rag-original-metadata.zip)을 내려받아 `.local/original-metadata`에 압축 해제합니다. 내부 `models/retriever` 폴더에는 `config.json`, `tokenizer.json`, `tokenizer_config.json`만 있어야 합니다. 설치된 프로젝트 환경에서 다음 오프라인 준비 명령을 실행합니다.

```powershell
.venv\Scripts\python.exe scripts\convert_original_retriever.py --checkpoint .local\original\week2_dense_retriever.pt --metadata .local\original-metadata\models\retriever --output .local\original-retriever
```

체크포인트의 SHA-256은 `ec09ae94fbdf5e8e86e1766353bcb71ea49aa222f4096f4f2ea2bfb65e6c33ca`와 일치해야 합니다. 스크립트는 체크포인트와 메타데이터를 먼저 검증한 뒤 CPU에서 `weights_only=True`로 복원합니다. 임의의 체크포인트를 허용하지 않으며 파일을 내려받지 않고 기존 출력도 보존합니다. 생성한 `model.safetensors`가 측정에 사용한 최초 모델의 해시 `2ce1e42b5a20ca387b3ba81eafb90bdeb7c73907ee6b66ebf9b1c7e17f9d681b` 및 크기 442,494,816 bytes와 정확히 일치해야 모델 폴더를 공개합니다. 변환 기록은 모델 폴더 옆에 저장합니다. 별도의 순수 CPU 환경에서 이 변환을 실제 수행해 기대한 바이트와 일치함을 확인했습니다. 모델 가중치를 준비하는 단계이므로 이 명령만으로 최초 모델의 비교 인덱스나 평가 점수까지 재현하지는 않습니다.

변환한 파일을 명시적으로 지정해 최초 모델의 비교용 묶음을 준비합니다.

```powershell
.venv\Scripts\python.exe scripts\download_artifacts.py --manifest reports\original-artifact-download.json --destination artifacts\original --local-retriever .local\original-retriever\model.safetensors
```

나머지 메타데이터와 공개 가중치를 기록된 출처에서 내려받고 원문을 사용자 환경에서 재구성한 뒤, 변환한 retriever를 선언된 `models/retriever/model.safetensors` 경로에만 복사합니다. SHA-256과 바이트 수가 최초 모델 manifest와 일치해야 합니다. 이미 내려받은 모델과의 충돌, 변경된 로컬 파일, 기존 출력 폴더는 덮어쓰지 않고 거부합니다. 최초 모델의 최종 비교 평가에는 `configs/original-freeze.json`을 사용하고 선택한 최종 모델에는 별도의 동결 기록을 사용합니다. 정확한 답변 불가 입력 구성과 출력 경로는 평가 보고서를 따릅니다.

## 자원 측정과 검증 범위

CPU 설치에는 모델, 가상환경, 임시 다운로드를 포함해 디스크 여유 공간 6 GB 이상을 실용적인 예산으로 권장합니다. 모든 환경에서 검증한 최소 사양은 아닙니다. 최종 선택 모델 묶음은 약 1.39 GB, 깨끗한 Windows CPU 가상환경은 약 0.98 GB입니다. CUDA 의존성, 보존한 환경 백업, 선택 사항인 최초 모델 비교 파일, 평가 결과에는 추가 공간이 필요합니다. 시스템 메모리 8 GB 이상은 여유를 확보하기 위한 권장이며 모든 실행 조건의 하한을 검증한 수치는 아닙니다.

최종 오프라인 검증에서는 같은 컴퓨터의 분리된 Windows와 Linux CPU 환경에서 사전에 지정한 개발 질문 두 개를 각각 두 번 실행했습니다. 동결된 소스 및 모델의 해시, 답변, 응답 여부, 인용 식별자가 일치했습니다. CPU 스레드 네 개를 사용했으며 프로세스 최대 메모리는 Windows 2.05 GiB, Linux 1.64 GiB였습니다. 별도의 초기 준비 검증에서는 Windows와 Linux의 자동 Python 준비와 CPU 의존성 설치도 실제 수행했습니다. 소규모 추론으로 확인한 실행 범위이며 전체 최종 평가나 외부 제3자의 독립 검증을 뜻하지 않습니다. 자세한 결과와 수치 차이는 [environment-verification.ko.md](../reports/environment-verification.ko.md)에 기록했습니다.
