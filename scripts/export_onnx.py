"""Export a PyTorch checkpoint to ONNX for later TensorRT conversion."""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    YOLO(args.weights).export(
        format="onnx",
        imgsz=640,
        batch=1,
        dynamic=False,
        simplify=True,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
