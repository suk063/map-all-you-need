"""Offline HTML explorer and scientific figures for exported map inputs."""

import json
import math
from pathlib import Path

import numpy as np


def render_report(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from plotly.offline import get_plotlyjs

    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    normalized = manifest.get('coordinate_frame') == 'normalized_robot'
    axis_labels = [f'{axis} (normalized)' if normalized else f'{axis} (m)' for axis in 'xyz']
    variants = manifest["variants"]
    payload = []
    for variant in variants:
        with np.load(output / variant["npz"]) as arrays:
            payload.append({**variant, "xyz": arrays["xyz"].round(6).tolist(),
                            "reset_xyz": arrays["reset_xyz"].round(6).tolist(),
                            "scores": arrays["scores"].round(6).tolist(), "colors": arrays["colors"].tolist(),
                            "feature_ids": arrays["feature_ids"].tolist(), "part_ids": arrays["part_ids"].tolist()})

    def draw(ax, item, individual=False):
        xyz = np.array(item["xyz"])
        rgb = np.array(item["colors"]) / 255
        ax.scatter(*xyz.T, c=rgb, s=4 if individual else 2, depthshade=False, linewidths=0, rasterized=True)
        center = (xyz.min(0) + xyz.max(0)) / 2
        radius = max(np.ptp(xyz, axis=0).max() * .54, .025)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
            axis.set_pane_color((.965, .975, .99, 1))
        ax.set(xlim=(center[0] - radius, center[0] + radius),
               ylim=(center[1] - radius, center[1] + radius),
               zlim=(center[2] - radius, center[2] + radius), xlabel=axis_labels[0], ylabel=axis_labels[1], zlabel=axis_labels[2])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-55)
        ax.tick_params(labelsize=7, pad=0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.label.set_size(8)
        ax.set_title(f"{item['task']}\n{item['points']:,} points · {len(item['parts'])} components", fontsize=11, pad=8)

    background = str(manifest["map_settings"]["background"]).lower()
    voxel_cm = manifest["map_settings"]["voxel_size"] * 100
    caption = (f"Seed {manifest['seed']} · first native step (zero action) · background={background} · voxel={voxel_cm:g} cm · all input points\n"
               "Frozen DINOv3 input (1024D): PC1 / PC2 / PC3 → R / G / B. PCA fitted separately per task/mode.")
    for mode in dict.fromkeys(item["robot"] for item in payload):
        chosen = [item for item in payload if item["robot"] == mode]
        rows = math.ceil(len(chosen) / 2)
        fig = plt.figure(figsize=(13, 3.9 * rows + 1.2), facecolor="white")
        for i, item in enumerate(chosen):
            draw(fig.add_subplot(rows, 2, i + 1, projection="3d"), item)
            single = plt.figure(figsize=(7, 6), facecolor="white")
            draw(single.add_subplot(projection="3d"), item, individual=True)
            single.text(.5, .035, f"robot={mode} · background={background} · seed={manifest['seed']} · first step (zero action)\nDINO input PCA → RGB", ha="center", fontsize=9)
            single.savefig(output / f"{item['task']}_{mode}.png", dpi=180, bbox_inches="tight")
            plt.close(single)
        fig.suptitle(f"Point Transformer inputs — robot={mode}", fontsize=20, y=.994)
        fig.text(.5, .963, caption, ha="center", va="top", fontsize=10)
        fig.subplots_adjust(top=.90, bottom=.025, left=.03, right=.97, hspace=.42, wspace=.08)
        fig.savefig(output / f"all_tasks_{mode}.png", dpi=160)
        plt.close(fig)
    template = Path(__file__).with_name("map_input_report.html").read_text()
    data = json.dumps({"manifest": manifest, "variants": payload}, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("/*__PLOTLY__*/", get_plotlyjs()).replace("/*__DATA__*/", data)
    (output / "index.html").write_text(html)
    print(f"Rendered {len(payload)} input clouds, contact sheets, and offline HTML", flush=True)
