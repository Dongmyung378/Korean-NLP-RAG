# Third-party attribution

[English](THIRD_PARTY_NOTICES.md) | [한국어](THIRD_PARTY_NOTICES.ko.md)

## Data

KorQuAD 1.0 is credited to its creators and source contributors. The [official dataset](https://korquad.github.io/category/1.0_KOR.html) declares [CC BY-ND 2.0 KR](https://creativecommons.org/licenses/by-nd/2.0/kr/deed.en). Public project archives contain IDs, hashes and reconstruction recipes rather than copied or transformed QA corpora. Setup downloads unchanged official source JSON and derives the runtime corpus and evaluation files locally.

## Models

| Component | Original source | Attribution and scope |
| --- | --- | --- |
| E5 dense encoder | [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3) | The pinned model card declares MIT. Credit the E5 authors and contributors. The project supplies required prefixes and indexes local documents; it does not claim authorship of this pretrained model. |
| Project retriever and reranker | [klue/roberta-base](https://huggingface.co/klue/roberta-base/tree/02f94ba5e3fcb7e2a58a390b8639b0fac974a8da) | Derived from KLUE-RoBERTa with project STS, relevance classification and question-passage refinement. Credit Sungjoon Park et al., KLUE-benchmark authors and contributors. |
| QA Reader | [monologg/koelectra-base-v3-finetuned-korquad](https://huggingface.co/monologg/koelectra-base-v3-finetuned-korquad/tree/ad4097ba1dc9904a22c3df71cafe6431e23f8914) | Credit monologg and upstream KoELECTRA contributors. Download unchanged weights from the recorded original repository. |

The [KLUE upstream repository](https://github.com/KLUE-benchmark/KLUE/tree/3efd98708a40ff49251fddde35453f8fbb11f536) releases the work under CC BY-SA 4.0. Its notice is preserved in [licenses/klue-cc-by-sa-4.0.md](licenses/klue-cc-by-sa-4.0.md). The pinned Hugging Face KLUE checkpoint has no separate license tag; applying the upstream declaration to model derivatives is an inference from that upstream work. Project fine-tuning changes are recorded in training reports.

The [KoELECTRA upstream repository](https://github.com/monologg/KoELECTRA/tree/024fbdd600e653b6e4bdfc64ceec84181b5ce6c4) and base discriminator declare Apache-2.0, preserved in [licenses/koelectra-apache-2.0.txt](licenses/koelectra-apache-2.0.txt). The exact pinned fine-tuned Reader repository does not supply a model card, license tag or license file. This project does not assign that checkpoint a separately verified license or rehost its model weights.

References: [KLUE paper](https://arxiv.org/abs/2105.09680), [E5 paper](https://arxiv.org/abs/2212.03533), [KoELECTRA project](https://github.com/monologg/KoELECTRA). Dependency packages retain their upstream licenses. This notice does not assign a new license to the project's own source code.
