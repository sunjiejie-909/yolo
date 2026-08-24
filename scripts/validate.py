"""Evaluate a checkpoint on an explicitly selected dataset partition."""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    YOLO(args.weights).val(
        data=args.data,
        split=args.split,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        plots=True,
    )


if __name__ == "__main__":
    main()
