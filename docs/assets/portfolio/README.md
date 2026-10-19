# Portfolio visuals

[English](README.md) | [한국어](README.ko.md)

The figures below use English labels. Each is a language-only derivative of the retained Korean SVG with the same chart geometry, palette and numerical values. The underlying experiment records and checksums remain in `visual-data.json`. Models, data splits, selection decisions and scientific reports are unchanged.

The localization renderer uses Python's standard library, the source SVGs and explicit labels in `localization.json`. It does not run models or repeat evaluation. Korean CLI questions and extracted Korean passages are task inputs and outputs, so their language is preserved rather than replaced with English examples of a different task.

```bash
python docs/assets/portfolio/render_localized_visuals.py
python docs/assets/portfolio/render_localized_visuals.py --check
```

`localization-receipt.json` binds each English derivative to the retained Korean source SVG and SHA-256. The original Korean renderer and `visual-receipt.json` remain unchanged. Diagrams describe processing steps rather than new measurements. Website assets and existing ZIPs are separate publication artifacts and are not overwritten by this helper.

## Figures

- [Retrieval and answer extraction](korean-rag-thumbnail.en.svg)
- [Paired answer and retrieval results](korean-rag-results.en.svg)
- [Runtime and false-answer tradeoffs](korean-rag-tradeoffs.en.svg)
