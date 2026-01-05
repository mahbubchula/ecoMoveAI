#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np


def draw_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle=None,
    facecolor="#FFFFFF",
    edgecolor="#0B3954",
    text_color="#1F2A35",
    subtitle_color="#3C4C58",
    linewidth=1.6,
    linestyle="solid",
):
    shadow = FancyBboxPatch(
        (x + 0.6, y - 0.6),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=2",
        linewidth=0,
        facecolor="#000000",
        alpha=0.08,
        zorder=2,
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=2",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=linestyle,
        zorder=3,
    )
    ax.add_patch(box)

    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=11.5,
            weight="bold",
            color=text_color,
            zorder=4,
        )
        ax.text(
            x + w / 2,
            y + h * 0.32,
            subtitle,
            ha="center",
            va="center",
            fontsize=9.2,
            color=subtitle_color,
            zorder=4,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=11.5,
            weight="bold",
            color=text_color,
            zorder=4,
        )


def draw_arrow(
    ax,
    start,
    end,
    color="#2E3A44",
    lw=1.4,
    style="-|>",
    mutation=14,
    connection="arc3,rad=0",
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle=connection,
        zorder=5,
    )
    ax.add_patch(arrow)


def center_left(box):
    x, y, w, h = box
    return (x, y + h / 2)


def center_right(box):
    x, y, w, h = box
    return (x + w, y + h / 2)


def center_top(box):
    x, y, w, h = box
    return (x + w / 2, y + h)


def center_bottom(box):
    x, y, w, h = box
    return (x + w / 2, y)


def build_figure(output_dir: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, ax = plt.subplots(figsize=(12.6, 7.4), dpi=160)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    cmap = LinearSegmentedColormap.from_list("bg", ["#F6F2ED", "#E8F0F7"])
    ax.imshow(
        gradient,
        extent=[0, 100, 0, 70],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        zorder=0,
    )

    for cx, cy, r, alpha in [(18, 64, 8, 0.08), (82, 12, 14, 0.06), (88, 56, 6, 0.07)]:
        ax.add_patch(
            Circle(
                (cx, cy),
                r,
                facecolor="#8FB6C9",
                edgecolor="none",
                alpha=alpha,
                zorder=1,
            )
        )

    ax.text(
        50,
        66.5,
        "EcoMoveAI System Architecture & Evaluation Pipeline",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
        color="#0B3954",
    )

    ax.text(17, 63.5, "Inputs", ha="center", va="center", fontsize=11, weight="bold", color="#35505E")
    ax.text(
        50,
        63.5,
        "Core Evaluation Engine",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color="#35505E",
    )
    ax.text(
        83,
        63.5,
        "Outputs & Interpretation",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color="#35505E",
    )

    core_panel = FancyBboxPatch(
        (34.5, 12.5),
        31,
        48.5,
        boxstyle="round,pad=0.02,rounding_size=4",
        linewidth=1.1,
        edgecolor="#C9D6DF",
        facecolor="#FFFFFF",
        linestyle=(0, (4, 4)),
        alpha=0.7,
        zorder=1.5,
    )
    ax.add_patch(core_panel)

    left_x, mid_x, right_x = 5, 38, 71
    box_w, mid_w, right_w = 24, 24, 24
    box_h, mid_h = 7, 7

    boxes = {
        "config": (left_x, 56, box_w, box_h),
        "knowledge": (left_x, 46, box_w, box_h),
        "memory": (left_x, 36, box_w, box_h),
        "main": (mid_x, 56, mid_w, mid_h),
        "baseline": (mid_x, 46, mid_w, mid_h),
        "ai": (mid_x, 36, mid_w, mid_h),
        "econ": (mid_x, 26, mid_w, mid_h),
        "results": (mid_x, 16, mid_w, mid_h),
        "reporting": (right_x, 44, right_w, 10),
        "streamlit": (right_x, 30, right_w, 7),
        "interpret": (right_x, 16, right_w, 10),
        "exports": (right_x, 6, right_w, 7),
    }

    draw_box(
        ax,
        *boxes["config"],
        title="Config JSON",
        subtitle="Baseline + AI rates\nPolicy scenarios",
        facecolor="#FDF2E6",
        edgecolor="#E09F3E",
    )
    draw_box(
        ax,
        *boxes["knowledge"],
        title="Knowledge Base",
        subtitle="Local RAG docs",
        facecolor="#FDF2E6",
        edgecolor="#E09F3E",
    )
    draw_box(
        ax,
        *boxes["memory"],
        title="Memory Notes",
        subtitle="Policy assumptions",
        facecolor="#FDF2E6",
        edgecolor="#E09F3E",
    )

    draw_box(
        ax,
        *boxes["main"],
        title="main.py",
        subtitle="Validation + orchestration",
        facecolor="#E7F0FA",
        edgecolor="#2F6690",
    )
    draw_box(
        ax,
        *boxes["baseline"],
        title="Baseline Model",
        subtitle="CO2 + fuel cost",
        facecolor="#E7F0FA",
        edgecolor="#2F6690",
    )
    draw_box(
        ax,
        *boxes["ai"],
        title="AI Scenarios",
        subtitle="Efficiency rates",
        facecolor="#E7F0FA",
        edgecolor="#2F6690",
    )
    draw_box(
        ax,
        *boxes["econ"],
        title="Economics Engine",
        subtitle="Policy instruments\nCosts + savings",
        facecolor="#E7F0FA",
        edgecolor="#2F6690",
    )
    draw_box(
        ax,
        *boxes["results"],
        title="Results Payload",
        subtitle="Scenario matrix",
        facecolor="#E7F0FA",
        edgecolor="#2F6690",
    )

    draw_box(
        ax,
        *boxes["reporting"],
        title="Reporting",
        subtitle="CSV + PNG assets",
        facecolor="#E8F5E9",
        edgecolor="#2E933C",
    )
    draw_box(
        ax,
        *boxes["streamlit"],
        title="Streamlit Dashboard",
        subtitle="Interactive explorer",
        facecolor="#E8F5E9",
        edgecolor="#2E933C",
    )
    draw_box(
        ax,
        *boxes["interpret"],
        title="Optional Interpretation",
        subtitle="LLM summary\nRAG + memory",
        facecolor="#F6EDEA",
        edgecolor="#B75D69",
        linestyle=(0, (4, 3)),
    )
    draw_box(
        ax,
        *boxes["exports"],
        title="Exports",
        subtitle="results.json/csv",
        facecolor="#E8F5E9",
        edgecolor="#2E933C",
    )

    draw_arrow(ax, center_right(boxes["config"]), center_left(boxes["main"]))

    draw_arrow(ax, center_bottom(boxes["main"]), center_top(boxes["baseline"]))
    draw_arrow(ax, center_bottom(boxes["baseline"]), center_top(boxes["ai"]))
    draw_arrow(ax, center_bottom(boxes["ai"]), center_top(boxes["econ"]))
    draw_arrow(ax, center_bottom(boxes["econ"]), center_top(boxes["results"]))

    draw_arrow(
        ax,
        center_right(boxes["results"]),
        center_left(boxes["reporting"]),
        connection="arc3,rad=0.2",
    )
    draw_arrow(ax, center_right(boxes["results"]), center_left(boxes["streamlit"]))
    draw_arrow(
        ax,
        center_right(boxes["results"]),
        center_left(boxes["interpret"]),
        connection="arc3,rad=-0.08",
    )
    draw_arrow(
        ax,
        center_right(boxes["results"]),
        center_left(boxes["exports"]),
        connection="arc3,rad=-0.2",
    )

    draw_arrow(
        ax,
        center_right(boxes["knowledge"]),
        center_left(boxes["interpret"]),
        connection="arc3,rad=-0.35",
    )
    draw_arrow(
        ax,
        center_right(boxes["memory"]),
        center_left(boxes["interpret"]),
        connection="arc3,rad=-0.2",
    )

    border = FancyBboxPatch(
        (1.5, 1.5),
        97,
        66.5,
        boxstyle="round,pad=0.02,rounding_size=6",
        linewidth=1.2,
        edgecolor="#CBD5DF",
        facecolor="none",
        zorder=6,
    )
    ax.add_patch(border)

    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / "eco_move_ai_system_diagram.svg"
    png_path = output_dir / "eco_move_ai_system_diagram.png"

    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.3)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    build_figure(repo_root / "figures")
