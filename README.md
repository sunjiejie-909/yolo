# RFGAC-YOLO Reproducibility Package

This private repository is an internal-review workspace for the manuscript "RFGAC-YOLO: Receptive-field global attention for UAV-based glass-insulator damage detection."

## Current status

The repository currently contains the verified dataset description, annotation protocol, experiment configuration, and release checklist. It does not yet contain field-inspection images, annotations, model source code, trained weights, or executable training scripts.

No reviewer-access or public reproducibility claim should be made until the items in `INTERNAL_RELEASE_CHECKLIST.md` have been completed.

## Verified GIDD summary

- Raw images collected: 2,753
- Images retained after screening: 2,542
- Classes: `insulator` and `defect`
- Pre-augmentation split: 1,779 training, 508 validation, 255 test
- Split ratio: 7:2:1
- Post-augmentation counts: 8,895 training, 2,540 validation, 255 test
- Test-set augmentation: none
- Annotation: two researchers independently annotated and cross-checked the samples; disagreements were resolved through discussion

## Planned public contents

- De-identified minimum reproducible GIDD subset
- Fixed `train.txt`, `val.txt`, and `test.txt` lists
- Dataset statistics and annotation protocol
- Model and training configuration files
- Training, validation, inference, and FPS-evaluation scripts
- Environment specification
- Reproducible model weights, subject to institutional approval

## Release policy

This repository must remain private during internal review. Before manuscript submission, the authors plan to:

1. complete the privacy and institutional review;
2. verify the minimum reproduction workflow;
3. make the repository public;
4. create a versioned GitHub release tied to a fixed commit; and
5. insert the public repository and release URLs into the manuscript's Data availability and Code availability statements.
