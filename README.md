# RFGAC-YOLO research code

This repository contains the code and configuration used to develop the
RFGAC-YOLO object detector for glass-insulator damage detection. The paper
authors are Hongjie Sun, Yanfeng Gao, Licong Guan, Yinglian Jin, and Binrui
Wang.

## Scope of this release

The public release contains:

- the required Ultralytics-based source tree;
- the custom layers in `ultralytics/nn/rfgac_modules.py`;
- model configurations in `configs/`;
- portable training, validation, prediction, export, and diagnostic scripts
  in `scripts/`;
- an example dataset configuration with placeholder paths;
- a CPU model-construction and forward-pass check in
  `scripts/smoke_test.py`.

The dataset, annotations, trained weights, TensorRT engines, `runs/` outputs,
logs, result tables, and other local experiment products are intentionally not
included. The example dataset file does not contain any private path or image.
The dataset may be released separately if the authors can do so under the
applicable data-use restrictions.

This repository is a code artifact. Its smoke test checks that the supplied
model configurations can be constructed and execute a forward pass; it does
not regenerate the numerical results reported in the paper without the
corresponding data, weights, and experiment records.

## Installation

Create a clean Python environment, install a PyTorch build appropriate for the
target system, and install this package in editable mode:

```bash
python -m pip install -e .
```

The custom module registration and parser handling are integrated in
`ultralytics/nn/tasks.py`. The code-level module names are retained for
compatibility with the model configuration files.

## Basic checks and scripts

Run the architecture and forward-pass check without downloading weights:

```bash
python scripts/smoke_test.py
python scripts/smoke_test.py --model configs/dscp_yolo_ablation.yaml
```

The training and validation scripts require a user-supplied dataset YAML and
weights or a training configuration. `scripts/benchmark_fps.py` measures a
model forward pass on a synthetic tensor only. It excludes preprocessing,
TensorRT execution, NMS, display, and other application stages, so its output
must not be interpreted as the paper's end-to-end edge-platform throughput.

## License and attribution

This project includes and modifies code from Ultralytics. The source and the
derivative project are distributed under the GNU Affero General Public License
v3.0. Please retain the attribution and license terms in `LICENSE` when
redistributing the code.

## Citation

Citation metadata for this code release is provided in `CITATION.cff`. No DOI,
archived release, dataset, or model checkpoint is claimed by this repository
until the authors create and verify one.
