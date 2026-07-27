#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize


def plot_ray_throughput_by_batch(ray_batches):
    fig, ax = plt.subplots()
    throughput_mrays = ray_batches["ray_throughput_rays_per_s"] / 1.0e6
    ax.scatter(ray_batches["batch_index"], throughput_mrays, s=6, alpha=0.5)
    ax.set_title("Ray Throughput by Batch")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Ray throughput (million rays/s)")
    fig.savefig("ray_throughput_by_batch.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_ray_throughput_by_unique_volumes_and_batch_size(ray_batches):
    fig, ax = plt.subplots()
    throughput_mrays = ray_batches["ray_throughput_rays_per_s"] / 1.0e6
    batch_size_mrays = ray_batches["num_rays"] / 1.0e6
    batch_size_norm = Normalize(batch_size_mrays.min(), batch_size_mrays.max())
    points = ax.scatter(ray_batches["num_unique_volumes"], throughput_mrays, c=batch_size_mrays, norm=batch_size_norm, s=10, alpha=0.5)
    ax.set_title("Ray Throughput by Unique Volumes and Batch Size")
    ax.set_xlabel("Unique volumes")
    ax.set_ylabel("Ray throughput (million rays/s)")
    fig.colorbar(points, ax=ax, label="Batch size (million rays)")
    fig.savefig("ray_throughput_by_unique_volumes_and_batch_size.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_ray_trace_time_by_batch_size_and_unique_volumes(ray_batches):
    fig, ax = plt.subplots()
    unique_volume_norm = Normalize(ray_batches["num_unique_volumes"].min(), ray_batches["num_unique_volumes"].max())
    points = ax.scatter(ray_batches["num_rays"] / 1.0e6, ray_batches["ray_trace_s"], c=ray_batches["num_unique_volumes"], norm=unique_volume_norm, s=10, alpha=0.5)
    ax.set_title("Ray Trace Time by Batch Size and Unique Volumes")
    ax.set_xlabel("Batch size (million rays)")
    ax.set_ylabel("Ray trace time (s)")
    fig.colorbar(points, ax=ax, label="Unique volumes")
    fig.savefig("ray_trace_time_by_batch_size_and_unique_volumes.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_unique_volumes_by_batch(ray_batches):
    fig, ax = plt.subplots()
    ax.scatter(ray_batches["batch_index"], ray_batches["num_unique_volumes"], s=6, alpha=0.5)
    ax.set_title("Unique Volumes by Batch")
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Unique volumes")
    fig.savefig("unique_volumes_by_batch.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_ray_batch_summary(ray_batches):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    throughput_mrays = ray_batches["ray_throughput_rays_per_s"] / 1.0e6
    batch_size_mrays = ray_batches["num_rays"] / 1.0e6
    batch_size_norm = Normalize(batch_size_mrays.min(), batch_size_mrays.max())

    axes[0].scatter(ray_batches["batch_index"], throughput_mrays, c=batch_size_mrays, norm=batch_size_norm, s=10, alpha=0.5)
    throughput_points = axes[1].scatter(ray_batches["num_unique_volumes"], throughput_mrays, c=batch_size_mrays, norm=batch_size_norm, s=10, alpha=0.5)

    axes[0].set_title("Ray Throughput by Batch")
    axes[0].set_xlabel("Batch index")
    axes[0].set_ylabel("Ray throughput (million rays/s)")

    axes[1].set_title("Ray Throughput by Unique Volumes")
    axes[1].set_xlabel("Unique volumes")
    axes[1].set_ylabel("Ray throughput (million rays/s)")

    fig.suptitle("Ray Batch Profiling Summary")
    fig.colorbar(throughput_points, ax=axes, label="Batch size (million rays)")
    fig.savefig("ray_batch_profile_summary.png", dpi=200, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Read and plot particle-simulation ray batch profiling data."
    )
    parser.add_argument("csv", help="Ray batch profiling CSV to read")
    args = parser.parse_args()
    ray_batches = pd.read_csv(args.csv)
    minimum_num_rays = 0.5 * ray_batches["num_rays"].max() # filter out queue-draining batches
    ray_batches = ray_batches[ray_batches["num_rays"] >= minimum_num_rays].copy()

    plot_ray_throughput_by_batch(ray_batches)
    plot_ray_throughput_by_unique_volumes_and_batch_size(ray_batches)
    plot_ray_trace_time_by_batch_size_and_unique_volumes(ray_batches)
    plot_unique_volumes_by_batch(ray_batches)
    plot_ray_batch_summary(ray_batches)

if __name__ == "__main__":
    main()
