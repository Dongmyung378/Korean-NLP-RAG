# Development retrieval selection

[한국어](development-retrieval.ko.md)

The frozen rule that gives Recall@5 priority selects public E5 with natural chunks of 192 tokens for the next Reader comparison. All six saved system comparisons use the same 304 answerable and 80 synthetic no-answer development questions, fixed corpus, original reranker, Reader using the top three chunks and original thresholds. Predictions, question bindings, component hashes and retrieval scores requiring the complete answer span were independently checked without new model inference.

| System | Recall@5 | EM | Character F1 | Answerable answered | False answers |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original development model | 267/304 (87.8289%) | 188/304 (61.8421%) | 68.9184% | 257/304 | 1/80 |
| Task-trained retriever, original chunks | 271/304 (89.1447%) | 189/304 (62.1711%) | 68.9616% | 257/304 | 5/80 |
| Public E5, chunks of 128 tokens | 289/304 (95.0658%) | 208/304 (68.4211%) | 75.6511% | 267/304 | 3/80 |
| Public E5, chunks of 192 tokens | 294/304 (96.7105%) | 222/304 (73.0263%) | 80.3362% | 282/304 | 4/80 |
| Public E5, chunks of 256 tokens | 293/304 (96.3816%) | 220/304 (72.3684%) | 80.8154% | 280/304 | 8/80 |
| Task-trained retriever, chunks of 192 tokens | 293/304 (96.3816%) | 224/304 (73.6842%) | 82.7348% | 283/304 | 11/80 |

The selected Recall@5 increases from 267/304 to 294/304, a gain of 8.8816 percentage points, exceeding the frozen one-point retrieval gate. EM increases from 188/304 to 222/304, or 11.1842 points. E5 candidates with 128, 192 and 256 tokens per chunk plus the task-trained candidate with 192 tokens exhaust the four configured chunking experiments.

The task-trained system with 192 tokens per chunk obtains two more exact answers than the selected candidate: EM 73.6842% versus 73.0263%, a 0.6579-point lead. It retrieves one fewer complete answer span in the top five and returns 11 unsupported synthetic answers versus 4. Task training produced useful development improvements; the chosen retrieval system follows the declared primary metric and preserves this tradeoff in the record.

This comparison changes the encoder, chunking and token budget together. It does not isolate the encoder contribution or establish statistical significance for a one-question ranking difference. E5 chunks of 256 tokens include prefix/special-token truncation under the actual tested dense budget.

This stage approves only the retrieval choice. Its raw 4/80 false-answer rate exceeds the unchanged GPU control 1/80, so full-pipeline promotion remains pending the fixed numerical guard, four Reader configurations and twelve threshold combinations. No final inference was performed. Exact fractions and evidence hashes are preserved in [the decision record](development-retrieval.json).
