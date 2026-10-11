# No-answer evaluation methodology

[한국어](no_answer_methodology.ko.md)

The new no-answer set tests whether the system rejects unsupported subject relations in the declared fixed knowledge base. It contains 80 development questions and 80 final questions, with 20 questions per subtype in each split: absent entity location, absent entity identity, fabricated relation and fictional future event.

Every synthetic question introduces a unique fictional referent that has no declared aliases. The referent is constructed solely for the benchmark, and no facts about it are inserted into the knowledge base. The question explicitly asks for that referent's property or relation. A normalization-aware scan checks its full name against every raw paragraph and title in the fixed corpus to reject accidental name collisions. The label rests on this controlled construction and required subject relation. Missing an answer string would not establish the label for a natural question.

Development questions use 50 source paragraphs from the frozen answerable development set. Final questions use 80 source paragraphs from the frozen answerable final set. Base paragraphs, complete fictional entity names and exact surface templates do not cross the split. Source articles and broad linguistic patterns can overlap. Construction does not run a model or inspect model errors. No final inference is allowed until the selected configuration, source, artifacts and both benchmark manifests are frozen.

The six historical manually selected probes remain a separate regression set. The fan-club dissolution case is unsupported by this fixed corpus: targeted review of all eight fan-club paragraphs and whole-corpus scans distinguish the source's 2014-03-29 founding ceremony from the requested dissolution event. The review records paragraph IDs, hashes and scope. It does not establish a universal claim about real-world events and is not independent human annotation. These exposed probes are excluded from the synthetic final denominator.

Report no-answer abstention and false-answer counts and rates by subtype, together with answerable coverage and accepted-answer exact match. The frozen selection protocol permits threshold promotion only when development no-answer false-answer rate is no worse than the original baseline, answerable coverage falls by no more than two percentage points, and the declared EM or F1 improvement rule is met. A 10% false-answer rate is an aspiration, not permission to lower coverage. When no candidate meets every rule, keep the incumbent.

Paired promotion uses the original baseline measured on the same device as the candidate. Tiny CPU/GPU logit differences can reverse a strict span-versus-CLS null tie. All four candidate Reader configurations therefore use one fixed `null_score_margin=1e-6` engineering guard, declared before configuration selection and never scanned for a better development score. The unchanged original control retains margin zero. The guard rejects near ties while preserving clear span decisions; it does not guarantee bitwise equivalence across hardware or eliminate every source of numerical variability.

This is a synthetic closed-corpus stress test. Fictional names and explicit future years may be easier to reject than natural ambiguous requests. It must not be described as a natural-query benchmark, an independent human review or proof of general hallucination prevention.

Recreate the records from locally supplied source manifests:

```powershell
python -m scripts.prepare_abstention --benchmark-dir PATH_TO_BENCHMARK --output-dir NEW_OUTPUT_DIRECTORY
```

The command refuses to overwrite existing output. The resulting manifest binds the source corpus, answerable benchmark, construction protocol and output files by SHA256.
