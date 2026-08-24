"""Run image or video inference without machine-specific paths."""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default="runs/predict")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    YOLO(args.weights).predict(
        source=args.source,
        imgsz=640,
        conf=args.conf,
        device=args.device,
        project=args.output,
        save=True,
    )


if __name__ == "__main__":
    main()
