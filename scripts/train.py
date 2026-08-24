"""Train RFGAC-YOLO with explicit, portable command-line arguments."""

import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--model", default="configs/rfgac_yolo.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/rfgac_yolo")
    parser.add_argument("--name", default="train")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        task="detect",
        imgsz=640,
        epochs=150,
        batch=4,
        workers=args.workers,
        device=args.device,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        mosaic=0.7,
        mixup=0.2,
        copy_paste=0.5,
        close_mosaic=0,
        patience=50,
        amp=True,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
