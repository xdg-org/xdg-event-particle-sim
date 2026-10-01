#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm, Normalize

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


def plot_filename(name, csv_stem, output_directory=None):
    filename = f"{csv_stem}_{name}.png"
    return Path(output_directory, filename) if output_directory else Path(filename)


def save_plot(fig, name, csv_stem, output_directory=None, *, tight=True):
    png_filename = plot_filename(name, csv_stem, output_directory)
    save_options = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(png_filename, dpi=300, **save_options)
    fig.savefig(png_filename.with_suffix(".pdf"), **save_options)


def process_ray_launches(ray_launches):
    if "event_type" not in ray_launches:
        ray_launches = ray_launches.copy()
        ray_launches["event_type"] = "unlabelled"

    processed_ray_launches = ray_launches.copy()
    processed_ray_launches["ray_throughput_mrays_per_s"] = processed_ray_launches["ray_throughput_rays_per_s"] / 1.0e6
    return processed_ray_launches


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
            "label": EVENT_TYPE_LABELS.get(
                event_type, event_type.replace("_", " ").title()
            ),
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


def plot_ray_throughput_by_launch(ray_launches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    scatter_by_event_type(
        ax,
        ray_launches["launch_index"],
        ray_launches["ray_throughput_mrays_per_s"],
        ray_launches["event_type"],
        size=8,
    )
    ax.set_title(plot_title("Ray Throughput by Launch", csv_title))
    ax.set_xlabel("Launch index")
    ax.set_ylabel("Ray throughput (million rays/s)")
    ax.legend(title="Event source")
    save_plot(fig, "ray_throughput_by_launch", csv_stem)
    plt.show()


def plot_ray_throughput_by_active_volumes_and_launch_size(ray_launches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    launch_size = ray_launches["num_rays"]
    launch_size_norm = LogNorm(launch_size.min(), launch_size.max())
    points = scatter_by_event_type(
        ax,
        ray_launches["num_active_volumes"],
        ray_launches["ray_throughput_mrays_per_s"],
        ray_launches["event_type"],
        color_values=launch_size,
        norm=launch_size_norm,
    )
    ax.set_title(plot_title("Ray Throughput by Active Volumes and Launch Size", csv_title))
    ax.set_xlabel("Active volumes")
    ax.set_ylabel("Ray throughput (million rays/s)")
    fig.colorbar(points, ax=ax, label="Launch size (rays, log scale)")
    ax.legend(title="Event source")
    save_plot(fig, "ray_throughput_by_active_volumes_and_launch_size", csv_stem)
    plt.show()


def plot_ray_trace_time_by_launch_size_and_active_volumes(ray_launches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    active_volume_norm = Normalize(ray_launches["num_active_volumes"].min(), ray_launches["num_active_volumes"].max())
    points = scatter_by_event_type(
        ax,
        ray_launches["num_rays"],
        ray_launches["ray_trace_s"],
        ray_launches["event_type"],
        color_values=ray_launches["num_active_volumes"],
        norm=active_volume_norm,
    )
    ax.set_title(plot_title("Ray Trace Time by Launch Size and Active Volumes", csv_title))
    ax.set_xlabel("Launch size (rays)")
    ax.set_ylabel("Ray trace time (s)")
    fig.colorbar(points, ax=ax, label="Active volumes")
    ax.legend(title="Event source")
    save_plot(fig, "ray_trace_time_by_launch_size_and_active_volumes", csv_stem)
    plt.show()


def plot_active_volumes_by_launch(ray_launches, csv_title, csv_stem):
    fig, ax = plt.subplots()
    scatter_by_event_type(
        ax,
        ray_launches["launch_index"],
        ray_launches["num_active_volumes"],
        ray_launches["event_type"],
        size=8,
    )
    ax.set_title(plot_title("Active Volumes by Launch", csv_title))
    ax.set_xlabel("Launch index")
    ax.set_ylabel("Active volumes")
    ax.legend(title="Event source")
    save_plot(fig, "active_volumes_by_launch", csv_stem)
    plt.show()


def plot_ray_launch_summary(
    ray_launches, csv_title, csv_stem, output_directory=None, show=True
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    launch_size = ray_launches["num_rays"]
    launch_size_norm = LogNorm(launch_size.min(), launch_size.max())

    scatter_by_event_type(
        axes[0],
        ray_launches["launch_index"],
        ray_launches["ray_throughput_mrays_per_s"],
        ray_launches["event_type"],
        color_values=launch_size,
        norm=launch_size_norm,
    )
    throughput_points = scatter_by_event_type(
        axes[1],
        ray_launches["num_active_volumes"],
        ray_launches["ray_throughput_mrays_per_s"],
        ray_launches["event_type"],
        color_values=launch_size,
        norm=launch_size_norm,
    )

    axes[0].set_title(plot_title("Ray Throughput by Launch", csv_title))
    axes[0].set_xlabel("Launch index")
    axes[0].set_ylabel("Ray throughput (million rays/s)")

    axes[1].set_title(plot_title("Ray Throughput by Active Volumes", csv_title))
    axes[1].set_xlabel("Active volumes")
    axes[1].set_ylabel("Ray throughput (million rays/s)")
    axes[1].legend(title="Event source")

    fig.suptitle(plot_title("Ray Launch Profiling Summary", csv_title))
    fig.colorbar(throughput_points, ax=axes, label="Launch size (rays, log scale)")
    save_plot(fig, "ray_launch_profile_summary", csv_stem, output_directory)
    if show:
        plt.show()
    else:
        plt.close(fig)


def select_volume_occupancy_launch(ray_launches, requested_launch_index):
    if requested_launch_index is not None:
        return requested_launch_index

    candidates = ray_launches.sort_values(
        ["num_active_volumes", "launch_index"], kind="stable"
    )
    half_ray_work = 0.5 * candidates["num_rays"].sum()
    return int(
        candidates.loc[
            candidates["num_rays"].cumsum() >= half_ray_work,
            "launch_index",
        ].iloc[0]
    )


def plot_volume_occupancy(
    volume_occupancies, ray_launches, csv_stem, requested_launch_index=None
):
    launch_index = select_volume_occupancy_launch(
        ray_launches, requested_launch_index
    )
    occupancy = volume_occupancies.query("launch_index == @launch_index")
    if occupancy.empty:
        raise ValueError(f"No volume occupancy data for launch {launch_index}")

    num_model_volumes = int(occupancy["num_model_volumes"].iloc[0])
    volume_ids = occupancy["volume_id"]
    counts = occupancy["num_rays"]
    expected_rays = int(
        ray_launches.set_index("launch_index").at[launch_index, "num_rays"]
    )
    if counts.sum() != expected_rays:
        raise ValueError(
            f"Volume occupancies for launch {launch_index} do not sum to its ray count"
        )

    fig, ax = plt.subplots(layout="constrained")
    ax.vlines(volume_ids, 1, counts, color="tab:blue", linewidth=0.8)
    for threshold, hardware, color, linestyle in (
        (32, "NVIDIA warp", "tab:orange", "--"),
        (64, "AMD wavefront", "tab:green", ":"),
    ):
        num_volumes = int((counts >= threshold).sum())
        ax.axhline(
            threshold,
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            label=f"≥{threshold} rays: {num_volumes:,} volumes ({hardware})",
        )

    ax.set(
        yscale="log",
        ylim=(1, None),
        xlabel="Volume ID",
        ylabel="Queued rays",
        title=(
            "Per-volume ray occupancy in a representative launch\n"
            f"{len(counts):,} of {num_model_volumes:,} volumes active"
        ),
    )
    ax.grid(axis="y", which="major", alpha=0.25)
    ax.legend(loc="upper right")
    save_plot(fig, "volume_occupancy", csv_stem)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Read and plot particle-simulation ray launch profiling data."
    )
    parser.add_argument("csv", help="Ray launch profiling CSV to read")
    parser.add_argument(
        "--volume-occupancy-csv",
        help="Long-format per-launch volume occupancy CSV to plot",
    )
    parser.add_argument(
        "--occupancy-launch-index",
        type=int,
        help=(
            "Launch to use for the individual occupancy plots; defaults to "
            "the ray-weighted median active-volume launch"
        ),
    )
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

    if args.volume_occupancy_csv:
        plot_volume_occupancy(
            pd.read_csv(args.volume_occupancy_csv),
            processed_ray_launches,
            csv_stem,
            args.occupancy_launch_index,
        )


if __name__ == "__main__":
    main()
