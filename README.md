# spacy_ner_pii

[![Release](https://github.com/irufano/spacy_ner_pii/actions/workflows/release.yml/badge.svg)](https://github.com/irufano/spacy_ner_pii/actions/workflows/release.yml)

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

The trained model is released separately as a wheel via [GitHub Releases](https://github.com/irufano/spacy_ner_pii/releases) (not committed to this repo) — grab the `.whl` URL from the latest release and install it directly:

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
# Budi Santoso                  PER
# Jl. Merdeka No. 10, Bandung   ADR
```

## Development

This repo contains the training pipeline, not the trained model artifacts (`pii_ner/models/`, `*.spacy` files are git-ignored and regenerated locally).

```bash
uv sync
```

- [`pii_ner/pii_data_train_generator.ipynb`](pii_ner/pii_data_train_generator.ipynb) — generates the synthetic labeled dataset (`pii_ner/dataset_ner_3000_v1.csv`, committed as the reproducible source of truth; the generator itself has no random seed, so don't regenerate it unless you intend to produce a new dataset version)
- [`pii_ner/pii_ner_train.ipynb`](pii_ner/pii_ner_train.ipynb) — converts the dataset to spaCy's binary format and fine-tunes `xx_ent_wiki_sm` (`pii_ner/config.cfg`)
- [`pii_ner/pii_ner_eval.ipynb`](pii_ner/pii_ner_eval.ipynb) — automatic dev-set benchmark + manual evaluation against the hand-labeled gold set
- [`pii_ner/scripts/build_docbin.py`](pii_ner/scripts/build_docbin.py) — standalone version of the CSV→`.spacy` conversion (used by CI, no Jupyter required)
- [`pii_ner/scripts/eval_gate.py`](pii_ner/scripts/eval_gate.py) — standalone version of the gold-set evaluation, exits non-zero if precision/recall drop below threshold (used by CI as a release gate)

## Release process

Pushing a semver tag (`vX.Y.Z`) to `main` triggers [`.github/workflows/release.yml`](.github/workflows/release.yml), which:

1. Rebuilds `train.spacy`/`dev.spacy` from the committed dataset CSV (`build_docbin.py`) — the dataset is **not** regenerated.
2. Fine-tunes `xx_ent_wiki_sm` from scratch using `pii_ner/config.cfg`.
3. Runs the gold-set eval gate (`eval_gate.py`) — the release is aborted if precision < 90% or recall < 97%.
4. Packages the model as a wheel (`spacy package ... --name ent_pii_sm`, which spaCy turns into `xx_ent_pii_sm-<version>`) and publishes it as a GitHub Release asset.
5. Bumps `pyproject.toml`/`uv.lock`'s version to match the tag and pushes that back to `main`.

A `workflow_dispatch` trigger is also available for dry runs (build + train + eval gate + package, without touching `main` or creating a release) before pushing a real tag.

## Known limitations

- Trained only on synthetic, templated sentences — while diverse, it does not cover every real-world phrasing.
- The base model's original `LOC`/`ORG`/`MISC` labels are not evaluated and likely degraded after fine-tuning (only `PER`/`ADR` were targeted).
- Optimized for Indonesian text; the `xx` (multilingual) base model gives partial coverage of other languages but this fine-tune has not been validated on them.

## License

[MIT](LICENSE)
