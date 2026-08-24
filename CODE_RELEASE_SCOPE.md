# Public code release scope

This repository is the code-only artifact for the RFGAC-YOLO manuscript.

## Included

- The model implementation and parser registration required by the supplied
  configurations.
- The RFGAC-YOLO and DSCP ablation configuration files.
- Portable training, validation, prediction, export, and diagnostic scripts.
- An example dataset configuration containing placeholders rather than local
  paths.
- Installation, licensing, and citation metadata.

## Excluded by design

- The GIDD and SFID image datasets and annotations.
- Trained checkpoints, ONNX files, TensorRT engines, and other model weights.
- `runs/`, logs, `results.csv`, benchmark records, screenshots, and generated
  figures.
- Local IDE settings, caches, private paths, internal audit notes, and old
  experiment documents.

The excluded material is not part of this commit and is not represented as
available from this repository. Dataset and supplementary artifact release can
be considered separately after the authors confirm data-use permissions and
the final public contents.

## Reproducibility boundary

`scripts/smoke_test.py` is a fast construction and forward-pass check. It does
not reproduce paper metrics. `scripts/benchmark_fps.py` measures model forward
time on a synthetic input and is not the paper's full application-pipeline FPS
measurement. No numerical result, checkpoint provenance, or hardware timing
record is generated or claimed by these scripts.
