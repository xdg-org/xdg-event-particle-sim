#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize


def plot_title(title, csv_title):
    return f"{title}: {csv_title}"


def plot_filename(name, csv_stem):
    return f"{csv_stem}_{name}.png"


def plot_ray_throughput_by_batch(ray_batches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    throughput_mrays = ray_batches["ray_throughput_rays_per_s"] / 1.0e6
    ax.scatter(ray_batches["batch_index"], throughput_mrays, s=6, alpha=0.5)
    ax.set_title(plot_title("Ray Throughput by Batch", csv_title))
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Ray throughput (million rays/s)")
    fig.savefig(plot_filename("ray_throughput_by_batch", csv_stem), dpi=200, bbox_inches="tight")
    plt.show()


def plot_ray_throughput_by_unique_volumes_and_batch_size(ray_batches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    throughput_mrays = ray_batches["ray_throughput_rays_per_s"] / 1.0e6
    batch_size_mrays = ray_batches["num_rays"] / 1.0e6
    batch_size_norm = Normalize(batch_size_mrays.min(), batch_size_mrays.max())
    points = ax.scatter(ray_batches["num_unique_volumes"], throughput_mrays, c=batch_size_mrays, norm=batch_size_norm, s=10, alpha=0.5)
    ax.set_title(plot_title("Ray Throughput by Unique Volumes and Batch Size", csv_title))
    ax.set_xlabel("Unique volumes")
    ax.set_ylabel("Ray throughput (million rays/s)")
    fig.colorbar(points, ax=ax, label="Batch size (million rays)")
    fig.savefig(plot_filename("ray_throughput_by_unique_volumes_and_batch_size", csv_stem), dpi=200, bbox_inches="tight")
    plt.show()


def plot_ray_trace_time_by_batch_size_and_unique_volumes(ray_batches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    unique_volume_norm = Normalize(ray_batches["num_unique_volumes"].min(), ray_batches["num_unique_volumes"].max())
    points = ax.scatter(ray_batches["num_rays"] / 1.0e6, ray_batches["ray_trace_s"], c=ray_batches["num_unique_volumes"], norm=unique_volume_norm, s=10, alpha=0.5)
    ax.set_title(plot_title("Ray Trace Time by Batch Size and Unique Volumes", csv_title))
    ax.set_xlabel("Batch size (million rays)")
    ax.set_ylabel("Ray trace time (s)")
    fig.colorbar(points, ax=ax, label="Unique volumes")
    fig.savefig(plot_filename("ray_trace_time_by_batch_size_and_unique_volumes", csv_stem), dpi=200, bbox_inches="tight")
    plt.show()


def plot_unique_volumes_by_batch(ray_batches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    ax.scatter(ray_batches["batch_index"], ray_batches["num_unique_volumes"], s=6, alpha=0.5)
    ax.set_title(plot_title("Unique Volumes by Batch", csv_title))
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Unique volumes")
    fig.savefig(plot_filename("unique_volumes_by_batch", csv_stem), dpi=200, bbox_inches="tight")
    plt.show()


def plot_ray_batch_summary(ray_batches, csv_title, csv_stem):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    throughput_mrays = ray_batches["ray_throughput_rays_per_s"] / 1.0e6
    batch_size_mrays = ray_batches["num_rays"] / 1.0e6
    batch_size_norm = Normalize(batch_size_mrays.min(), batch_size_mrays.max())

    axes[0].scatter(ray_batches["batch_index"], throughput_mrays, c=batch_size_mrays, norm=batch_size_norm, s=10, alpha=0.5)
    throughput_points = axes[1].scatter(ray_batches["num_unique_volumes"], throughput_mrays, c=batch_size_mrays, norm=batch_size_norm, s=10, alpha=0.5)

    axes[0].set_title(plot_title("Ray Throughput by Batch", csv_title))
    axes[0].set_xlabel("Batch index")
    axes[0].set_ylabel("Ray throughput (million rays/s)")

    axes[1].set_title(plot_title("Ray Throughput by Unique Volumes", csv_title))
    axes[1].set_xlabel("Unique volumes")
    axes[1].set_ylabel("Ray throughput (million rays/s)")

    fig.suptitle(plot_title("Ray Batch Profiling Summary", csv_title))
    fig.colorbar(throughput_points, ax=axes, label="Batch size (million rays)")
    fig.savefig(plot_filename("ray_batch_profile_summary", csv_stem), dpi=200, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Read and plot particle-simulation ray batch profiling data."
    )
    parser.add_argument("csv", help="Ray batch profiling CSV to read")
    args = parser.parse_args()
    csv_path = Path(args.csv)
    csv_title = csv_path.name
    csv_stem = csv_path.stem
    ray_batches = pd.read_csv(args.csv)
    minimum_num_rays = 0.5 * ray_batches["num_rays"].max() # filter out queue-draining batches
    ray_batches = ray_batches[ray_batches["num_rays"] >= minimum_num_rays].copy()

    plot_ray_throughput_by_batch(ray_batches, csv_title, csv_stem)
    plot_ray_throughput_by_unique_volumes_and_batch_size(ray_batches, csv_title, csv_stem)
    plot_ray_trace_time_by_batch_size_and_unique_volumes(ray_batches, csv_title, csv_stem)
    plot_unique_volumes_by_batch(ray_batches, csv_title, csv_stem)
    plot_ray_batch_summary(ray_batches, csv_title, csv_stem)

if __name__ == "__main__":
    main()
