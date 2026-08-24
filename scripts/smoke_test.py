"""Build the model and execute one forward pass without downloading weights."""

import argparse

import torch

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="configs/rfgac_yolo.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.model.to(args.device).eval()
    sample = torch.zeros(1, 3, 640, 640, device=args.device)
    with torch.inference_mode():
        outputs = model.model(sample)
    print(f"model={args.model}")
    print(f"input_shape={tuple(sample.shape)}")
    print(f"output_type={type(outputs).__name__}")
    print("smoke_test=PASS")


if __name__ == "__main__":
    main()
