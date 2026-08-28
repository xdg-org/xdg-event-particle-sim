#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    fig.savefig(plot_filename("ray_throughput_by_launch", csv_stem), dpi=200, bbox_inches="tight")
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
    fig.savefig(plot_filename("ray_throughput_by_active_volumes_and_launch_size", csv_stem), dpi=200, bbox_inches="tight")
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
    fig.savefig(plot_filename("ray_trace_time_by_launch_size_and_active_volumes", csv_stem), dpi=200, bbox_inches="tight")
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
    fig.savefig(plot_filename("active_volumes_by_launch", csv_stem), dpi=200, bbox_inches="tight")
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
    fig.savefig(
        plot_filename("ray_launch_profile_summary", csv_stem, output_directory),
        dpi=200,
        bbox_inches="tight",
    )
    if show:
        plt.show()
    else:
        plt.close(fig)


def process_volume_occupancies(volume_occupancies, ray_launches):
    required_columns = {
        "launch_index",
        "num_model_volumes",
        "volume_id",
        "num_rays",
    }
    missing_columns = required_columns.difference(volume_occupancies.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Volume occupancy CSV is missing columns: {missing}")

    launch_columns = ["launch_index", "num_rays", "event_type"]
    source_count_columns = [
        "num_initial_rays",
        "num_collision_rays",
        "num_surface_crossing_rays",
    ]
    launch_columns.extend(
        column for column in source_count_columns if column in ray_launches.columns
    )
    launch_metadata = ray_launches[launch_columns].rename(
        columns={"num_rays": "num_launch_rays"}
    )
    processed = volume_occupancies.merge(
        launch_metadata,
        on="launch_index",
        how="left",
        validate="many_to_one",
    )
    if processed["num_launch_rays"].isna().any():
        raise ValueError("Volume occupancy CSV contains an unknown launch index")

    occupancy_totals = processed.groupby("launch_index")["num_rays"].sum()
    launch_totals = launch_metadata.set_index("launch_index")["num_launch_rays"]
    mismatched = occupancy_totals[occupancy_totals != launch_totals.loc[occupancy_totals.index]]
    if not mismatched.empty:
        launch_index = int(mismatched.index[0])
        raise ValueError(
            f"Volume occupancies for launch {launch_index} do not sum to its ray count"
        )

    return processed


def ray_work_median_launch(ray_launches):
    ordered = ray_launches.sort_values("launch_index")
    cumulative_rays = ordered["num_rays"].cumsum()
    median_position = 0.5 * ordered["num_rays"].sum()
    return int(ordered.loc[cumulative_rays >= median_position, "launch_index"].iloc[0])


def select_occupancy_launch(ray_launches, requested_launch_index=None):
    if requested_launch_index is None:
        return ray_work_median_launch(ray_launches)
    if requested_launch_index not in set(ray_launches["launch_index"]):
        raise ValueError(f"Launch {requested_launch_index} is not present in the profiling data")
    return requested_launch_index


def launch_volume_counts(volume_occupancies, launch_index):
    launch = volume_occupancies[
        volume_occupancies["launch_index"] == launch_index
    ]
    if launch.empty:
        raise ValueError(f"No volume occupancy data found for launch {launch_index}")

    num_model_volumes = int(launch["num_model_volumes"].iloc[0])
    if not (launch["num_model_volumes"] == num_model_volumes).all():
        raise ValueError(f"Inconsistent model volume counts for launch {launch_index}")
    volume_ids = launch["volume_id"].to_numpy()
    counts = launch["num_rays"].to_numpy(dtype=np.int64)
    inactive_volumes = num_model_volumes - len(counts)
    if inactive_volumes < 0:
        raise ValueError(f"Launch {launch_index} has more occupied volumes than the model")
    return volume_ids, counts, inactive_volumes


def draw_occupancy_histogram(ax, volume_ids, counts):
    ax.vlines(volume_ids, 0, counts, color="tab:blue", linewidth=0.8)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Volume ID")
    ax.set_ylabel("Queued rays")
    ax.grid(axis="y", alpha=0.25)


def launch_description(ray_launches, launch_index):
    launch = ray_launches[ray_launches["launch_index"] == launch_index].iloc[0]
    event_label = EVENT_TYPE_LABELS.get(
        launch["event_type"], str(launch["event_type"]).replace("_", " ").title()
    )
    return f"Launch {launch_index}: {event_label}, {int(launch['num_rays']):,} rays"


def plot_volume_occupancy_histogram(
    volume_occupancies, ray_launches, launch_index, csv_title, csv_stem
):
    volume_ids, counts, inactive_volumes = launch_volume_counts(
        volume_occupancies, launch_index
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_occupancy_histogram(ax, volume_ids, counts)
    ax.set_title(plot_title("Volume Occupancy by Volume ID", csv_title))
    ax.text(
        0.99,
        0.97,
        launch_description(ray_launches, launch_index),
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    fig.savefig(
        plot_filename("volume_occupancy_histogram", csv_stem),
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


def plot_ranked_volume_occupancy(
    volume_occupancies, ray_launches, launch_index, csv_title, csv_stem
):
    _, counts, inactive_volumes = launch_volume_counts(volume_occupancies, launch_index)
    ranked_counts = np.sort(counts)[::-1]
    ranks = np.arange(1, len(ranked_counts) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ranks, ranked_counts, color="tab:blue")
    ax.set_title(plot_title("Ranked Volume Occupancy", csv_title))
    ax.set_xlabel("Occupied volume rank")
    ax.set_ylabel("Queued rays")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.99,
        0.97,
        f"{launch_description(ray_launches, launch_index)}\n"
        f"Inactive volumes: {inactive_volumes:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    fig.savefig(
        plot_filename("ranked_volume_occupancy", csv_stem),
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


def plot_cumulative_volume_coverage(
    volume_occupancies, ray_launches, launch_index, csv_title, csv_stem
):
    _, counts, inactive_volumes = launch_volume_counts(volume_occupancies, launch_index)
    ranked_counts = np.sort(counts)[::-1]
    cumulative_fraction = np.cumsum(ranked_counts) / ranked_counts.sum()
    ranks = np.arange(1, len(ranked_counts) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ranks, cumulative_fraction, color="tab:blue")
    ax.axhline(0.5, color="tab:gray", linestyle="--", linewidth=1)
    ax.axhline(0.9, color="tab:gray", linestyle="--", linewidth=1)
    ax.set_ylim(0.0, 1.01)
    ax.set_title(plot_title("Cumulative Ray Coverage by Volume", csv_title))
    ax.set_xlabel("Highest-occupancy volumes included")
    ax.set_ylabel("Fraction of queued rays")
    ax.grid(alpha=0.25)
    ax.text(
        0.99,
        0.03,
        f"{launch_description(ray_launches, launch_index)}\n"
        f"Inactive volumes: {inactive_volumes:,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    fig.savefig(
        plot_filename("cumulative_volume_coverage", csv_stem),
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()


def source_representative_launch(ray_launches, count_column):
    if count_column not in ray_launches or ray_launches[count_column].max() <= 0:
        return None
    fractions = ray_launches[count_column] / ray_launches["num_rays"]
    candidates = ray_launches.assign(source_fraction=fractions).sort_values(
        ["source_fraction", "launch_index"], ascending=[False, True]
    )
    return int(candidates.iloc[0]["launch_index"])


def selected_occupancy_launches(ray_launches, representative_launch_index):
    candidates = [
        ("Initial", int(ray_launches["launch_index"].min())),
        (
            "Collision-heavy",
            source_representative_launch(ray_launches, "num_collision_rays"),
        ),
        (
            "Surface-crossing-heavy",
            source_representative_launch(
                ray_launches, "num_surface_crossing_rays"
            ),
        ),
        ("Ray-work median", representative_launch_index),
        ("Final", int(ray_launches["launch_index"].max())),
    ]
    selected = []
    seen = set()
    for label, launch_index in candidates:
        if launch_index is not None and launch_index not in seen:
            selected.append((label, launch_index))
            seen.add(launch_index)
    return selected


def plot_selected_launch_occupancies(
    volume_occupancies,
    ray_launches,
    representative_launch_index,
    csv_title,
    csv_stem,
):
    selected = selected_occupancy_launches(
        ray_launches, representative_launch_index
    )
    num_columns = min(3, len(selected))
    num_rows = math.ceil(len(selected) / num_columns)
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(5 * num_columns, 4 * num_rows),
        squeeze=False,
        layout="constrained",
    )
    for ax, (selection_label, launch_index) in zip(axes.flat, selected):
        volume_ids, counts, inactive_volumes = launch_volume_counts(
            volume_occupancies, launch_index
        )
        draw_occupancy_histogram(ax, volume_ids, counts)
        ax.set_title(
            f"{selection_label}\n{launch_description(ray_launches, launch_index)}"
        )
    for ax in list(axes.flat)[len(selected):]:
        ax.set_visible(False)

    fig.suptitle(plot_title("Selected Launch Volume Occupancies", csv_title))
    fig.savefig(
        plot_filename("selected_launch_volume_occupancies", csv_stem),
        dpi=200,
        bbox_inches="tight",
    )
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
        help="Launch to use for the individual occupancy plots; defaults to the ray-work median",
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
        raw_volume_occupancies = pd.read_csv(args.volume_occupancy_csv)
        volume_occupancies = process_volume_occupancies(
            raw_volume_occupancies, processed_ray_launches
        )
        launch_index = select_occupancy_launch(
            processed_ray_launches, args.occupancy_launch_index
        )
        occupancy_title = Path(args.volume_occupancy_csv).name
        plot_volume_occupancy_histogram(
            volume_occupancies,
            processed_ray_launches,
            launch_index,
            occupancy_title,
            csv_stem,
        )
        plot_ranked_volume_occupancy(
            volume_occupancies,
            processed_ray_launches,
            launch_index,
            occupancy_title,
            csv_stem,
        )
        plot_cumulative_volume_coverage(
            volume_occupancies,
            processed_ray_launches,
            launch_index,
            occupancy_title,
            csv_stem,
        )
        plot_selected_launch_occupancies(
            volume_occupancies,
            processed_ray_launches,
            launch_index,
            occupancy_title,
            csv_stem,
        )


if __name__ == "__main__":
    main()
