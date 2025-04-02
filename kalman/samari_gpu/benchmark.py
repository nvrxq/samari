import matplotlib.pyplot as plt
import numpy as np
import time
import logging
import torch
from kalman_filter import KalmanFilter
from kalman_samurai import KalmanFilter as KalmanFilterSamurai
import argparse
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_tensor(bs):
    return torch.randn(bs, 4, dtype=torch.float)


def plot_results(results: dict):
    """
    Plot benchmark results comparing KalmanFilter implementations

    :args:
    results: {
        "bs": list[int],        # Batch sizes
        "time": list[float],    # Execution times in seconds for our implementation
        "time_samurai": list[float], # Execution times in seconds for Samurai implementation
    }

    X-axis: Batch size (number of samples)
    Y-axis: Time (seconds)
    """
    plt.figure(figsize=(12, 6))
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.plot(
        results["bs"],
        results["time"],
        label="KalmanFilterSamari(ours)",
        linewidth=2.5,
        marker="o",
        markersize=6,
        linestyle="-",
        color="#2ecc71",
    )

    plt.plot(
        results["bs"],
        results["time_samurai"],
        label="KalmanFilterSamurai",
        linewidth=2.5,
        marker="s",
        markersize=6,
        linestyle="--",
        color="#e74c3c",
    )

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Batch size", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title("Kalman Filter Implementation Comparison", fontsize=14, pad=15)
    plt.legend(fontsize=10, frameon=True, facecolor="white", edgecolor="gray")

    plt.tight_layout()
    plt.savefig("benchmark.png", dpi=300, bbox_inches="tight")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-bs", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    bs = range(50, args.max_bs + 1)
    time_samari = []
    time_samurai = []

    for b in tqdm(bs):
        tensor = get_tensor(b)
        start = time.time()
        samari_filter = KalmanFilter()
        mean, cov = samari_filter.initiate(tensor.to(device="cuda"))
        mean_pred, cov_pred = samari_filter.predict(mean, cov)
        time_samari.append(time.time() - start)

        start = time.time()
        samurai_filter = KalmanFilterSamurai()
        for i in range(tensor.shape[0]):
            mean, cov = samurai_filter.initiate(tensor[i])
            mean_pred, cov_pred = samurai_filter.predict(mean, cov)
        time_samurai.append(time.time() - start)

    plot_results(
        {
            "bs": bs,
            "time": time_samari,
            "time_samurai": time_samurai,
        }
    )


if __name__ == "__main__":
    main()
