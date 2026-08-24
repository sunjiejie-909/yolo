"""Measure synchronized model-forward latency on a synthetic input tensor."""

import argparse
import statistics
import time

import numpy as np
import torch

from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import select_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--half", action="store_true")
    args = parser.parse_args()

    device = select_device(args.device)
    model = attempt_load_weights(args.weights, device=device, fuse=True).eval()
    sample = torch.randn(1, 3, 640, 640, device=device)
    if args.half:
        model = model.half()
        sample = sample.half()

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        timings = []
        for _ in range(args.iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)

    mean_seconds = statistics.fmean(timings)
    print(f"mean_ms={mean_seconds * 1000:.4f}")
    print(f"median_ms={statistics.median(timings) * 1000:.4f}")
    print(f"p95_ms={np.percentile(timings, 95) * 1000:.4f}")
    print(f"fps={1.0 / mean_seconds:.4f}")
    print("scope=model forward only; excludes decode, preprocessing, NMS, drawing, and display")


if __name__ == "__main__":
    main()
