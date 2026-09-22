# DAFR core code

This folder contains a compact reference implementation of the architecture and enrollment protocol described in the manuscript, **DAFR: Dual-Space Adaptive Forensic Representation Learning for AI-Generated Image Detection and Attribution**.

## Code map

| Manuscript component | Code |
| --- | --- |
| Low-bit, directional-residual, and spectral views | `dafr/forensic_views.py` |
| Four ResNet-18 encoders and reliability-guided fusion | `dafr/model.py` |
| Asymmetric authenticity/source spaces | `dafr/model.py` |
| Reliability alignment and cross-space decorrelation | `dafr/objectives.py` |
| Compact few-shot parameter updates | `dafr/adaptation.py`, `enroll.py` |
| Reference refresh and instance-level source ranking | `dafr/registry.py` |
| Base training and inference entry points | `train_base.py`, `predict.py` |

The RGB encoder conditions the gate and is not a fourth forensic expert. During adaptation, the four encoders, fusion, and prediction heads are frozen. After each update, the registry re-encodes all retained source references with the current model. The binary head handles real/fake detection; real images receive no source attribution.

## Setup

Install the packages in `requirements.txt`. Image paths in the CSV manifests may be absolute or relative to the manifest file. Each CSV uses this schema:

```csv
path,binary_label,source
images/real_001.png,0,
images/sd21_001.png,1,Stable Diffusion 2.1
```

The training and validation manifests must have the same source labels for the base generator classes.

## Base training

```bash
python train_base.py --train data/train.csv --validation data/validation.csv --checkpoint checkpoints/base.pth
```

The entry point uses the manuscript's 10 epochs, batch size 24, and separate encoder/downstream learning rates by default. It also supports real-only base initialization when the training manifest contains no synthetic source rows.

## Enrollment and inference

To create a registry for base-training sources without changing the model, prepare a CSV of retained reference images and run:

```bash
python enroll.py --checkpoint checkpoints/base.pth --references data/base_references.csv --steps 0 --output-checkpoint checkpoints/base_registered.pth --output-registry checkpoints/registry.json
```

To add a new generator, supply its labeled references and authentic anchors. The existing registry supplies retained exemplars from previously enrolled sources:

```bash
python enroll.py --checkpoint checkpoints/base_registered.pth --registry checkpoints/registry.json --references data/new_references.csv --anchors data/real_anchors.csv --output-checkpoint checkpoints/updated.pth --output-registry checkpoints/updated_registry.json
```

The registry JSON stores paths to reference images. Keep those files available; the next enrollment or inference call refreshes their embeddings with the current model.

```bash
python predict.py --checkpoint checkpoints/updated.pth --registry checkpoints/updated_registry.json --image data/query.png
```
