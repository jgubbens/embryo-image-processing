# Computer Vision Analysis of Drosophila Embryo

Pipeline for staging *Drosophila* embryo development (nuclear cycles NC9–NC14+) from live-imaging microscopy video, plus the preprocessing and training code that supports it.

## Pipeline

1. **ND2 to TIFF conversion** ([src/processing/convert_nd2.py](src/processing/convert_nd2.py))
   Splits a multi-position, multi-channel `.nd2` acquisition into per-position, per-channel ImageJ-compatible TIFF stacks, with optional spatial binning and calibration metadata preserved.

2. **Embryo extraction** ([src/processing/extract_embryo.py](src/processing/extract_embryo.py))
   Segments the embryo from each frame with a Cellpose model, then rotates/centers/scales it into a canonical 800x800 frame so downstream classifiers see a consistent orientation and scale.

3. **Nuclear cycle classification** ([src/classification/](src/classification/))
   - [nc10_classification.py](src/classification/nc10_classification.py) — binary ResNet18 classifier for pre- vs. post-NC10.
   - [train_separated_classifiers.py](src/classification/train_separated_classifiers.py) — trains one binary ResNet18 classifier per NC transition (NC9 → NC14+).
   - [hmm_design3/](src/classification/hmm_design3/) — the main staging model: a ResNet18 CNN (optionally feeding an LSTM, see [design.md](src/classification/hmm_design3/design.md)) produces per-frame nuclear-cycle probabilities, which are combined with a learned state-duration model and tracked over time via a forward algorithm / online HMM (`hmm_classifier.py`, `hmm_predictor.py`, `forward_algorithm.py`) for live, frame-by-frame state prediction.

4. **Augmentation** ([src/augmentation/](src/augmentation/))
   Elastic deformation and other augmentation utilities used when training data is limited.

## Project layout

```
src/
  processing/      ND2 conversion, embryo segmentation/extraction, histogram equalization
  augmentation/     elastic deformation and embryo image augmentation
  classification/   NC-stage classifiers and the CNN+HMM live predictor (hmm_design3/)
data/                training/inference data (gitignored, not included)
models/              trained model checkpoints and duration model (gitignored, not included)
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```