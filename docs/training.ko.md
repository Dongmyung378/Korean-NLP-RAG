# 제한된 예산의 모델 학습

[English](training.md) | [한국어](training.ko.md)

기존 노트북 실험을 재현 가능한 질의와 문서 청크의 검색 학습 및 무작위 음성 샘플과 어려운 음성 샘플을 비교하는 재순위화 학습으로 구성했습니다. 학습에는 공식 KorQuAD 학습 질문만 사용합니다. 프로젝트 개발 세트와 최종 세트의 질문은 학습 데이터 구성, 음성 샘플 추출, 체크포인트 선택에 사용하지 않습니다.

## 초기 모델과 환경

[실행 안내](runtime.ko.md)에 따라 환경을 준비합니다. 학습은 선택 사항이며 CPU에서는 시간이 크게 늘어날 수 있습니다. 기록된 실험 환경은 PyTorch 2.14.0+cu130, Transformers 5.17.0, 메모리 6 GB의 RTX 3060 Laptop GPU입니다.

원래 프로젝트에서 학습한 검색 모델과 재순위화 모델 및 각각의 토크나이저를 `.local/training-inputs/retriever`와 `.local/training-inputs/reranker`에 준비합니다. 추가 개선 이전의 모델이 초기값이며, 최종 실행 모델로 대체하면 같은 실험을 재현할 수 없습니다. 공개된 아티팩트 명세와 다음 해시를 확인합니다.

선택 자료인 [학습 모델 압축 파일](https://drive.google.com/file/d/1attqOUViSx2i4q0JcRz2l577HZTGIuh6/view?usp=drivesdk)을 내려받아 [압축 파일 안내](../reports/training-artifacts.ko.md)의 크기와 SHA-256을 확인한 뒤 `models/original-retriever`를 `.local/training-inputs/retriever`에 복사합니다. 함께 제공하는 `models/task-retriever-hard-epoch1`은 기록된 실험 후보이며 같은 실험을 다시 학습할 때 초기 모델로 사용하지 않습니다. 원래 재순위화 모델은 별도로 준비한 실행 구성의 `artifacts/runtime/models/reranker`에서 `.local/training-inputs/reranker`로 복사합니다. 최종 질의응답 구성은 이 압축의 검색 모델 대신 E5를 사용합니다.

| 입력 파일 | SHA-256 |
| --- | --- |
| 원래 `week2_dense_retriever.pt` | `ec09ae94fbdf5e8e86e1766353bcb71ea49aa222f4096f4f2ea2bfb65e6c33ca` |
| 변환한 원래 검색 모델 `model.safetensors` | `2ce1e42b5a20ca387b3ba81eafb90bdeb7c73907ee6b66ebf9b1c7e17f9d681b` |
| 원래 재순위화 모델 `model.safetensors` | `98071cd9882265c172865279b91f75e0b40c6f5e9787be60296ee82fb5f29d2c` |
| 원래 KLUE 구조 `config.json` | `71e0b0ea43d6b05c1a92484e0deb8a71b471f7787c3a7c97fc664e62347a08f5` |

구조 설정의 출처는 `klue/roberta-base` 리비전 `02f94ba5e3fcb7e2a58a390b8639b0fac974a8da`입니다. 12개 어텐션 헤드, 12개 계층, 은닉 차원 768을 명시합니다. 텐서의 크기만으로 어텐션 헤드 수를 추정하지 않고 원래 설정을 제공해 가중치를 엄격하게 불러옵니다. 토크나이저와 초기 모델 파일의 해시는 변환 명세 및 학습 명세에 기록합니다.

검색 모델이 변환된 Hugging Face 디렉터리가 아닌 기존 체크포인트로 제공된 경우 한 번 변환합니다. 실제 압축 파일 구성에 맞게 입력 경로 세 개를 조정합니다.

```powershell
.venv\Scripts\python.exe -m scripts.train_models convert `
  --initial-model .local\training-inputs\week2_dense_retriever.pt `
  --tokenizer .local\training-inputs\tokenizer `
  --architecture-config .local\training-inputs\klue-roberta-base-config.json `
  --output-dir .local\training-inputs\retriever
```

이후 모든 출력 디렉터리도 새 경로여야 합니다. 스크립트는 이전 결과를 덮어쓰지 않습니다.

## 데이터 재구성과 학습 분할

```powershell
.venv\Scripts\python.exe -m scripts.reproduce_data `
  --source-dir .local\source-data --output-dir .local\benchmark-v1

.venv\Scripts\python.exe -m scripts.prepare_training_data build `
  --train-source .local\benchmark-v1\train.jsonl `
  --tokenizer .local\training-inputs\retriever `
  --output-dir .local\training-data `
  --train-questions 3000 --dev-questions 200 `
  --chunk-max-tokens 256 --overlap-tokens 32 --seed 378
```

공식 원본 파일을 그대로 내려받고 공개된 질문 ID 명세로 측정 당시의 벤치마크를 재구성합니다. 원본 다운로드의 해시, 분할별 개수, 생성된 JSONL 파일 다섯 개의 해시가 모두 일치해야 합니다. 학습 구성은 `train.jsonl`만 읽고 정규화 후 중복 질문을 제거하며, 고정 시드로 문단 단위 그룹을 선택합니다. 내부 개발 문단은 학습 문단이나 학습 음성 샘플 풀과 겹치지 않습니다. 문단을 분리하지 않고 보존하므로 실제 구성은 학습 질문 3,002개와 내부 개발 질문 206개입니다.

청크 경계는 문장 경계와 토큰 길이로 정하며 정답 위치를 이용하지 않습니다. 학습 양성 청크만 완전한 정답 범위를 포함하는지 확인해 선택합니다. 기록된 데이터는 학습 문단 478개와 청크 697개, 내부 개발 문단 34개와 청크 52개이며 완전한 정답 범위가 없어 제외된 선택 질문은 없습니다.

## 같은 질문으로 음성 샘플 비교

```powershell
.venv\Scripts\python.exe -m scripts.prepare_training_data mine `
  --data-dir .local\training-data `
  --initial-model .local\training-inputs\retriever `
  --output-dir .local\mining `
  --device cuda --max-length 64 --top-k 50 --batch-size 16 --seed 378
```

CUDA가 없으면 `--device cpu`를 사용합니다. 기존 프로젝트의 STS 검색 모델을 고정하고, 마스킹된 평균 풀링과 L2 정규화로 각 분할의 코퍼스를 따로 검색합니다. 정답 문단, 같은 문서 제목, 정규화한 주석 정답을 포함하는 청크를 음성 샘플에서 제외합니다. 알려진 거짓 음성 샘플을 걸러내지만 정답 주석이 불완전한 경우의 모호성까지 제거했다고 주장하지 않습니다.

무작위 방식과 어려운 음성 샘플 방식은 질문 ID, 양성 청크, 예제 수가 같습니다. 어느 한쪽의 유효한 음성 샘플이 없으면 양쪽에서 함께 제외합니다. 기록된 실험은 학습 3,002개와 내부 개발 206개 질문을 모두 유지했습니다. 추출과 구성 명세에 모델 출처, 필터링 결과, 입력 및 출력 해시를 기록합니다.

## 검색 모델 및 재순위화 모델 학습

```powershell
.venv\Scripts\python.exe -m scripts.train_models train `
  --task retriever `
  --train .local\mining\train-hard-triplets.json `
  --dev .local\mining\dev-hard-triplets.json `
  --development-corpus .local\training-data\dev-corpus.jsonl `
  --initial-model .local\training-inputs\retriever `
  --config configs\training.json `
  --output-dir .local\runs\task-retriever-hard --device cuda

.venv\Scripts\python.exe -m scripts.train_models train `
  --task reranker `
  --train .local\mining\train-random-triplets.json `
  --dev .local\mining\dev-hard-triplets.json `
  --initial-model .local\training-inputs\reranker `
  --config configs\training.json `
  --output-dir .local\runs\reranker-random --device cuda

.venv\Scripts\python.exe -m scripts.train_models train `
  --task reranker `
  --train .local\mining\train-hard-triplets.json `
  --dev .local\mining\dev-hard-triplets.json `
  --initial-model .local\training-inputs\reranker `
  --config configs\training.json `
  --output-dir .local\runs\reranker-hard --device cuda
```

재순위화 비교의 두 실험은 같은 기존 프로젝트 모델에서 시작합니다. 이 모델은 이미 어려운 음성 샘플로 학습한 이력이 있으므로 비교의 의미는 무작위 방식과 어려운 음성 샘플 방식의 추가 학습 효과입니다. 사전학습 언어 모델부터 각각 학습하는 독립 비교와는 구분해야 합니다. 과거 재순위화 학습에 사용한 질문 ID 목록이 보존되지 않아 초기 모델이 내부 개발 예제를 과거에 접했는지는 확인할 수 없습니다.

`configs/training.json`은 시드 378, 학습률 1e-5, 가중치 감쇠 0.01, 최대 2에포크, 마이크로배치 2, 8회 기울기 누적을 고정합니다. 검색 모델의 질의와 청크 길이는 각각 128토큰과 256토큰이고 온도는 0.05입니다. 재순위화 모델의 질의와 청크 쌍은 최대 384토큰입니다. AdamW, 10% 선형 워밍업, 기울기 제한, 체크포인팅, 선택적 CUDA 혼합 정밀도를 사용합니다. 마지막 부분 배치를 포함해 실제 샘플 수에 비례하도록 기울기를 누적합니다.

검색 학습은 각 질의를 해당 양성 청크 및 명시적으로 선택한 음성 청크와 비교합니다. 배치의 다른 양성 청크를 일괄적으로 음성으로 취급하지 않습니다. 체크포인트 선택 점수는 검색 모델의 내부 Recall@5 + 0.01 × MRR@20이며, 재순위화 모델은 고정된 내부 개발 쌍에서 양성 점수가 어려운 음성 샘플 점수보다 높은 비율을 사용합니다.

학습 전 초기 모델의 점수를 평가하고 `best_model`로 저장합니다. 선택 점수가 0.001을 초과해 개선될 때만 체크포인트를 교체하며, 개선되지 않는 첫 에포크에서 중단합니다. 최대 예산은 2에포크입니다. 개선이 없으면 `selected_epoch`는 0이고 초기 가중치를 유지합니다. 각 출력에는 `training_report.json`, `input_manifest.json`, 이식 가능한 `best_model` 디렉터리가 포함됩니다.

## 기록된 검색 학습 결과와 평가 범위

질의와 청크를 대상으로 추가 학습한 검색 모델은 1에포크 체크포인트를 선택했습니다. 2에포크에서는 성능이 떨어져 실험을 중단했습니다. 아래 수치는 내부 개발 질문 206개와 청크 52개에 대한 결과입니다.

| 내부 학습 지표 | 초기 모델 | 선택한 1에포크 | 제외한 2에포크 |
| --- | ---: | ---: | ---: |
| Recall@1 | 84.95% | 93.20% | 91.75% |
| Recall@5 | 96.12% | 100.00% | 99.03% |
| Recall@20 | 99.03% | 100.00% | 100.00% |
| MRR@20 | 0.90275 | 0.96197 | 0.95449 |

376회 최적화에 790.85초가 걸렸으며 PyTorch의 최대 CUDA 할당량은 2,232,149,504바이트였습니다. 이 지표와 재순위화의 쌍별 순위 정확도는 체크포인트 선택을 위한 진단 수치입니다. 최종 RAG 정답률이나 대규모 미관측 코퍼스에서의 일반화 성능을 의미하지 않습니다. 실행 모델로 채택하려면 고정된 프로젝트 개발 세트에서 별도로 RAG 성능을 비교해야 합니다. 최종 질문은 모델, 리더, 응답 정책을 확정한 뒤 평가하며 실제 채택 구성과 최종 성능은 공개 평가 보고서에 기록합니다.

재순위화 추가 학습 비교에서는 사전에 정한 선택 기준에 따른 개선이 없었습니다.

| 추가 학습 방식 | 초기 쌍별 순위 정확도 | 1에포크 정확도 | 선택 에포크 | 최적화 횟수 | 실행 시간 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 무작위 음성 샘플 | 100.00% | 100.00% | 0 | 188 | 136.19초 |
| 어려운 음성 샘플 | 100.00% | 100.00% | 0 | 188 | 138.24초 |

두 실험 모두 1에포크 후 중단했고 초기 가중치를 유지했습니다. 선택 가중치의 SHA-256은 위 표의 `98071cd9...`와 바이트 단위로 같습니다. 초기 정확도 100%는 작은 고정 쌍 평가의 상한입니다. 전체 검색 후보 목록을 항상 올바르게 정렬한다는 뜻이나 어느 음성 샘플 방식이 일반적으로 우수하다는 뜻은 아닙니다. 정체가 확인되어 추가 에포크나 학습 변형을 진행하지 않았습니다. PyTorch 최대 CUDA 할당량은 무작위 방식 2,251,112,960바이트, 어려운 음성 샘플 방식 2,251,657,728바이트였습니다.

다른 장치나 소프트웨어 빌드에서 다시 학습하면 수치가 달라질 수 있습니다. 입력 해시와 시드는 실험의 추적 가능성을 제공하며 모든 환경에서 동일한 부동소수점 가중치를 보장하지 않습니다.
