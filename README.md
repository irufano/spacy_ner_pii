# spacy_ner_pii

A [spaCy](https://spacy.io) NER pipeline fine-tuned to detect **PII (Personally Identifiable Information)** in free-form Indonesian text — currently covering:

- **`PER`** — person names (single names, compound names, titles/honorifics, international names)
- **`ADR`** — addresses (Indonesian street format, kost/dusun/perumahan/apartemen, PO Box, and international formats)

The model is fine-tuned from spaCy's official multilingual [`xx_ent_wiki_sm`](https://github.com/explosion/spacy-models) pipeline on a synthetically generated dataset (no real PII was used for training).

## Status

Evaluated on a 105-sentence hand-labeled gold set covering hard cases (dialog attribution, enumerations, ALL-CAPS headlines, form/table text, social media handles, name/place disambiguation):

| Label | Precision | Recall | F1 |
|---|---|---|---|
| `PER` | 95.5% | 97.7% | 96.6% |
| `ADR` | 100% | 100% | 100% |
| **Overall** | **96.4%** | **98.2%** | **97.3%** |

## Installation

The trained model is released separately as a wheel via [GitHub Releases](https://github.com/irufano/spacy_ner_pii/releases) (not committed to this repo):

```bash
pip install https://github.com/irufano/spacy_ner_pii/releases/download/v0.1.0/xx_ent_pii_sm-0.1.0-py3-none-any.whl
```

## Usage

```python
import xx_ent_pii_sm

nlp = xx_ent_pii_sm.load()
doc = nlp("Budi Santoso tinggal di Jl. Merdeka No. 10, Bandung.")

for ent in doc.ents:
    print(ent.text, ent.label_)
# Budi Santoso            PER
# Jl. Merdeka No. 10, Bandung   ADR
```

## Development

This repo contains the training pipeline, not the trained model artifacts (`pii_ner/models/`, `*.spacy` files are git-ignored and regenerated locally).

```bash
uv sync
```

- [`pii_ner/pii_data_train_generator.ipynb`](pii_ner/pii_data_train_generator.ipynb) — generates the synthetic labeled dataset (`dataset_ner_3000_v*.csv`)
- [`pii_ner/pii_ner_train.ipynb`](pii_ner/pii_ner_train.ipynb) — converts the dataset to spaCy's binary format and fine-tunes `xx_ent_wiki_sm` (`pii_ner/config.cfg`)
- [`pii_ner/pii_ner_eval.ipynb`](pii_ner/pii_ner_eval.ipynb) — automatic dev-set benchmark + manual evaluation against the hand-labeled gold set

## Known limitations

- Trained only on synthetic, templated sentences — while diverse, it does not cover every real-world phrasing.
- The base model's original `LOC`/`ORG`/`MISC` labels are not evaluated and likely degraded after fine-tuning (only `PER`/`ADR` were targeted).
- Optimized for Indonesian text; the `xx` (multilingual) base model gives partial coverage of other languages but this fine-tune has not been validated on them.

## License

[MIT](LICENSE)
