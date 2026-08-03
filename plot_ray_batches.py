#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize


EVENT_TYPE_ORDER = ["initial", "collision", "surface_crossing", "mixed", "unlabelled"]
EVENT_TYPE_LABELS = {
    "initial": "Initial",
    "collision": "Collision",
    "surface_crossing": "Surface crossing",
    "mixed": "Mixed",
    "unlabelled": "Unlabelled",
}
EVENT_TYPE_MARKERS = {
    "initial": "*",
    "collision": "o",
    "surface_crossing": "^",
    "mixed": "s",
    "unlabelled": "o",
}
EVENT_TYPE_COLORS = {
    "initial": "tab:orange",
    "collision": "tab:blue",
    "surface_crossing": "tab:green",
    "mixed": "tab:red",
    "unlabelled": "tab:gray",
}


def plot_title(title, csv_title):
    return f"{title}: {csv_title}"


def plot_filename(name, csv_stem):
    return f"{csv_stem}_{name}.png"


def process_ray_batches(ray_batches):
    if "event_type" not in ray_batches:
        ray_batches = ray_batches.copy()
        ray_batches["event_type"] = "unlabelled"

    # Filter out queue-draining batches before deriving plot-ready columns.
    minimum_num_rays = 0.5 * ray_batches["num_rays"].max()
    processed_ray_batches = ray_batches[ray_batches["num_rays"] >= minimum_num_rays].copy()

    # Add derived metrics to the dataframe
    processed_ray_batches["ray_throughput_mrays_per_s"] = processed_ray_batches["ray_throughput_rays_per_s"] / 1.0e6
    processed_ray_batches["batch_size_mrays"] = processed_ray_batches["num_rays"] / 1.0e6
    return processed_ray_batches


def scatter_by_event_type(
    ax,
    x,
    y,
    event_types,
    *,
    color_values=None,
    norm=None,
    size=10,
    alpha=0.5,
):
    first_collection = None
    present_event_types = list(dict.fromkeys(event_types))
    ordered_event_types = [
        event_type
        for event_type in EVENT_TYPE_ORDER
        if event_type in present_event_types
    ]
    ordered_event_types.extend(
        event_type
        for event_type in present_event_types
        if event_type not in EVENT_TYPE_ORDER
    )

    for event_type in ordered_event_types:
        mask = event_types == event_type
        scatter_kwargs = {
            "marker": EVENT_TYPE_MARKERS.get(event_type, "x"),
            "label": EVENT_TYPE_LABELS.get(event_type, event_type.replace("_", " ").title()),
            "s": size,
            "alpha": alpha,
        }
        if color_values is None:
            scatter_kwargs["color"] = EVENT_TYPE_COLORS.get(event_type, "tab:gray")
        else:
            scatter_kwargs["c"] = color_values[mask]
            scatter_kwargs["cmap"] = "viridis"
            scatter_kwargs["norm"] = norm

        collection = ax.scatter(x[mask], y[mask], **scatter_kwargs)
        if first_collection is None:
            first_collection = collection

    return first_collection


def plot_ray_throughput_by_batch(ray_batches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    scatter_by_event_type(
        ax,
        ray_batches["batch_index"],
        ray_batches["ray_throughput_mrays_per_s"],
        ray_batches["event_type"],
        size=8,
    )
    ax.set_title(plot_title("Ray Throughput by Batch", csv_title))
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Ray throughput (million rays/s)")
    ax.legend(title="Event source")
    fig.savefig(plot_filename("ray_throughput_by_batch", csv_stem), dpi=200, bbox_inches="tight")
    plt.show()


def plot_ray_throughput_by_unique_volumes_and_batch_size(ray_batches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    batch_size_mrays = ray_batches["batch_size_mrays"]
    batch_size_norm = Normalize(batch_size_mrays.min(), batch_size_mrays.max())
    points = scatter_by_event_type(
        ax,
        ray_batches["num_unique_volumes"],
        ray_batches["ray_throughput_mrays_per_s"],
        ray_batches["event_type"],
        color_values=batch_size_mrays,
        norm=batch_size_norm,
    )
    ax.set_title(plot_title("Ray Throughput by Unique Volumes and Batch Size", csv_title))
    ax.set_xlabel("Unique volumes")
    ax.set_ylabel("Ray throughput (million rays/s)")
    fig.colorbar(points, ax=ax, label="Batch size (million rays)")
    ax.legend(title="Event source")
    fig.savefig(plot_filename("ray_throughput_by_unique_volumes_and_batch_size", csv_stem), dpi=200, bbox_inches="tight")
    plt.show()


def plot_ray_trace_time_by_batch_size_and_unique_volumes(ray_batches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    unique_volume_norm = Normalize(ray_batches["num_unique_volumes"].min(), ray_batches["num_unique_volumes"].max())
    points = scatter_by_event_type(
        ax,
        ray_batches["batch_size_mrays"],
        ray_batches["ray_trace_s"],
        ray_batches["event_type"],
        color_values=ray_batches["num_unique_volumes"],
        norm=unique_volume_norm,
    )
    ax.set_title(plot_title("Ray Trace Time by Batch Size and Unique Volumes", csv_title))
    ax.set_xlabel("Batch size (million rays)")
    ax.set_ylabel("Ray trace time (s)")
    fig.colorbar(points, ax=ax, label="Unique volumes")
    ax.legend(title="Event source")
    fig.savefig(plot_filename("ray_trace_time_by_batch_size_and_unique_volumes", csv_stem), dpi=200, bbox_inches="tight")
    plt.show()


def plot_unique_volumes_by_batch(ray_batches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    scatter_by_event_type(
        ax,
        ray_batches["batch_index"],
        ray_batches["num_unique_volumes"],
        ray_batches["event_type"],
        size=8,
    )
    ax.set_title(plot_title("Unique Volumes by Batch", csv_title))
    ax.set_xlabel("Batch index")
    ax.set_ylabel("Unique volumes")
    ax.legend(title="Event source")
    fig.savefig(plot_filename("unique_volumes_by_batch", csv_stem), dpi=200, bbox_inches="tight")
    plt.show()


def plot_ray_batch_summary(ray_batches, csv_title, csv_stem):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    batch_size_mrays = ray_batches["batch_size_mrays"]
    batch_size_norm = Normalize(batch_size_mrays.min(), batch_size_mrays.max())

    scatter_by_event_type(
        axes[0],
        ray_batches["batch_index"],
        ray_batches["ray_throughput_mrays_per_s"],
        ray_batches["event_type"],
        color_values=batch_size_mrays,
        norm=batch_size_norm,
    )
    throughput_points = scatter_by_event_type(
        axes[1],
        ray_batches["num_unique_volumes"],
        ray_batches["ray_throughput_mrays_per_s"],
        ray_batches["event_type"],
        color_values=batch_size_mrays,
        norm=batch_size_norm,
    )

    axes[0].set_title(plot_title("Ray Throughput by Batch", csv_title))
    axes[0].set_xlabel("Batch index")
    axes[0].set_ylabel("Ray throughput (million rays/s)")

    axes[1].set_title(plot_title("Ray Throughput by Unique Volumes", csv_title))
    axes[1].set_xlabel("Unique volumes")
    axes[1].set_ylabel("Ray throughput (million rays/s)")
    axes[1].legend(title="Event source")

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
    processed_ray_batches = process_ray_batches(ray_batches)

    plot_ray_throughput_by_batch(processed_ray_batches, csv_title, csv_stem)
    plot_ray_throughput_by_unique_volumes_and_batch_size(processed_ray_batches, csv_title, csv_stem)
    plot_ray_trace_time_by_batch_size_and_unique_volumes(processed_ray_batches, csv_title, csv_stem)
    plot_unique_volumes_by_batch(processed_ray_batches, csv_title, csv_stem)
    plot_ray_batch_summary(processed_ray_batches, csv_title, csv_stem)


if __name__ == "__main__":
    main()
