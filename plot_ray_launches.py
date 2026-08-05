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


def process_ray_launches(ray_launches):
    # Filter out queue-draining launches before deriving plot-ready columns.
    minimum_num_rays = 0.5 * ray_launches["num_rays"].max()
    processed_ray_launches = ray_launches[ray_launches["num_rays"] >= minimum_num_rays].copy()

    # Add derived metrics to the dataframe
    processed_ray_launches["ray_throughput_mrays_per_s"] = processed_ray_launches["ray_throughput_rays_per_s"] / 1.0e6
    processed_ray_launches["launch_size_mrays"] = processed_ray_launches["num_rays"] / 1.0e6
    return processed_ray_launches


def plot_ray_throughput_by_launch(ray_launches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    ax.scatter(ray_launches["launch_index"], ray_launches["ray_throughput_mrays_per_s"], s=6, alpha=0.5)
    ax.set_title(plot_title("Ray Throughput by Launch", csv_title))
    ax.set_xlabel("Launch index")
    ax.set_ylabel("Ray throughput (million rays/s)")
    fig.savefig(plot_filename("ray_throughput_by_launch", csv_stem), dpi=200, bbox_inches="tight")
    plt.show()


def plot_ray_throughput_by_active_volumes_and_launch_size(ray_launches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    launch_size_mrays = ray_launches["launch_size_mrays"]
    launch_size_norm = Normalize(launch_size_mrays.min(), launch_size_mrays.max())
    points = ax.scatter(ray_launches["num_active_volumes"], ray_launches["ray_throughput_mrays_per_s"], c=launch_size_mrays, norm=launch_size_norm, s=10, alpha=0.5)
    ax.set_title(plot_title("Ray Throughput by Active Volumes and Launch Size", csv_title))
    ax.set_xlabel("Active volumes")
    ax.set_ylabel("Ray throughput (million rays/s)")
    fig.colorbar(points, ax=ax, label="Launch size (million rays)")
    fig.savefig(plot_filename("ray_throughput_by_active_volumes_and_launch_size", csv_stem), dpi=200, bbox_inches="tight")
    plt.show()


def plot_ray_trace_time_by_launch_size_and_active_volumes(ray_launches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    active_volume_norm = Normalize(ray_launches["num_active_volumes"].min(), ray_launches["num_active_volumes"].max())
    points = ax.scatter(ray_launches["launch_size_mrays"], ray_launches["ray_trace_s"], c=ray_launches["num_active_volumes"], norm=active_volume_norm, s=10, alpha=0.5)
    ax.set_title(plot_title("Ray Trace Time by Launch Size and Active Volumes", csv_title))
    ax.set_xlabel("Launch size (million rays)")
    ax.set_ylabel("Ray trace time (s)")
    fig.colorbar(points, ax=ax, label="Active volumes")
    fig.savefig(plot_filename("ray_trace_time_by_launch_size_and_active_volumes", csv_stem), dpi=200, bbox_inches="tight")
    plt.show()


def plot_active_volumes_by_launch(ray_launches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    ax.scatter(ray_launches["launch_index"], ray_launches["num_active_volumes"], s=6, alpha=0.5)
    ax.set_title(plot_title("Active Volumes by Launch", csv_title))
    ax.set_xlabel("Launch index")
    ax.set_ylabel("Active volumes")
    fig.savefig(plot_filename("active_volumes_by_launch", csv_stem), dpi=200, bbox_inches="tight")
    plt.show()


def plot_ray_launch_summary(ray_launches, csv_title, csv_stem):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    launch_size_mrays = ray_launches["launch_size_mrays"]
    launch_size_norm = Normalize(launch_size_mrays.min(), launch_size_mrays.max())

    axes[0].scatter(ray_launches["launch_index"], ray_launches["ray_throughput_mrays_per_s"], c=launch_size_mrays, norm=launch_size_norm, s=10, alpha=0.5)
    throughput_points = axes[1].scatter(ray_launches["num_active_volumes"], ray_launches["ray_throughput_mrays_per_s"], c=launch_size_mrays, norm=launch_size_norm, s=10, alpha=0.5)

    axes[0].set_title(plot_title("Ray Throughput by Launch", csv_title))
    axes[0].set_xlabel("Launch index")
    axes[0].set_ylabel("Ray throughput (million rays/s)")

    axes[1].set_title(plot_title("Ray Throughput by Active Volumes", csv_title))
    axes[1].set_xlabel("Active volumes")
    axes[1].set_ylabel("Ray throughput (million rays/s)")

    fig.suptitle(plot_title("Ray Launch Profiling Summary", csv_title))
    fig.colorbar(throughput_points, ax=axes, label="Launch size (million rays)")
    fig.savefig(plot_filename("ray_launch_profile_summary", csv_stem), dpi=200, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Read and plot particle-simulation ray launch profiling data."
    )
    parser.add_argument("csv", help="Ray launch profiling CSV to read")
    args = parser.parse_args()
    csv_path = Path(args.csv)
    csv_title = csv_path.name
    csv_stem = csv_path.stem
    ray_launches = pd.read_csv(args.csv)
    processed_ray_launches = process_ray_launches(ray_launches)

    plot_ray_throughput_by_launch(processed_ray_launches, csv_title, csv_stem)
    plot_ray_throughput_by_active_volumes_and_launch_size(processed_ray_launches, csv_title, csv_stem)
    plot_ray_trace_time_by_launch_size_and_active_volumes(processed_ray_launches, csv_title, csv_stem)
    plot_active_volumes_by_launch(processed_ray_launches, csv_title, csv_stem)
    plot_ray_launch_summary(processed_ray_launches, csv_title, csv_stem)


if __name__ == "__main__":
    main()
