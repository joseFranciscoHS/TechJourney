#!/usr/bin/env python3
"""Render Hybrid RGS backbone architecture schematics (Methods-faithful).

Matches paper/figures/prompts/arch_*.md topology/labels. Outputs PNG+SVG
under paper/figures/ for LaTeX includegraphics. Replace with Paper Banana
exports using the same filenames if preferred.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "paper" / "figures"

C_BLOCK = "#D6EAF8"
C_PROC = "#FDEBD0"
C_DEC = "#D5F5E3"
C_LATENT = "#E8DAEF"
C_FILM = "#FCF3CF"
C_EDGE = "#2C3E50"
C_SKIP = "#7F8C8D"
C_NOTE = "#34495E"


def _setup(figsize=(4.2, 9.5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def _box(ax, x, y, w, h, text, facecolor=C_BLOCK, fontsize=8):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1,
        edgecolor=C_EDGE,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=C_EDGE,
        linespacing=1.15,
    )
    return (x, y - h / 2), (x, y + h / 2)


def _arrow(ax, p_from, p_to, color=C_EDGE, lw=1.2, connectionstyle="arc3,rad=0"):
    ax.add_patch(
        FancyArrowPatch(
            p_from,
            p_to,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def _vchain(ax, nodes, x=5.0):
    centers, bottoms, tops = [], [], []
    for y, h, text, color, fs in nodes:
        bot, top = _box(ax, x, y, 5.6, h, text, facecolor=color, fontsize=fs)
        centers.append((x, y))
        bottoms.append(bot)
        tops.append(top)
    for i in range(len(nodes) - 1):
        _arrow(ax, bottoms[i], (bottoms[i][0], tops[i + 1][1] + 0.02))
    return centers


def _save(fig, path: Path) -> None:
    fig.tight_layout(pad=0.3)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_drcnet3d(path: Path) -> None:
    fig, ax = _setup((4.0, 10.2))
    ax.set_ylim(0, 22)
    ax.text(
        5,
        21.4,
        "DRCNet3D Architecture",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=C_EDGE,
    )
    nodes = [
        (20.2, 0.85, "Input\n$K$ ch", C_BLOCK, 8),
        (18.9, 0.75, "Conv3d  $3{\\times}3{\\times}3$", C_BLOCK, 7.5),
        (17.6, 0.9, "Conv3d  $2{\\times}2{\\times}2$\nstride 2", C_BLOCK, 7.5),
        (16.3, 0.65, "FiLM (optional)", C_FILM, 7.5),
        (
            14.5,
            1.5,
            "RCDB  $\\times 4$ iter\nGRU gates $u_t$, $r_t$\nFactorized $3{\\times}1{\\times}1$,\n$1{\\times}3{\\times}1$, $1{\\times}1{\\times}3$",
            C_PROC,
            7,
        ),
        (12.7, 0.9, "ConvTranspose3d\n$2{\\times}2{\\times}2$ stride 2", C_BLOCK, 7.5),
        (11.4, 0.65, "FiLM (optional)", C_FILM, 7.5),
        (10.1, 0.7, "Concat w/ skip", C_BLOCK, 7.5),
        (8.9, 0.6, "Conv3d  $1{\\times}1{\\times}1$", C_BLOCK, 7.5),
        (7.8, 0.6, "Conv3d  $3{\\times}3{\\times}3$", C_BLOCK, 7.5),
        (6.7, 0.65, "PReLU / Sigmoid", C_BLOCK, 7.5),
        (5.5, 0.75, "Output\n$1$ ch", C_BLOCK, 8),
    ]
    centers = _vchain(ax, nodes, x=4.2)
    ax.text(
        7.85,
        16.3,
        "$[\\cos_x,\\cos_y,$\n$\\cos_z,b_{\\mathrm{norm}}]$",
        fontsize=6.5,
        color=C_NOTE,
        ha="left",
        va="center",
    )
    ax.plot(
        [centers[1][0] + 2.8, centers[1][0] + 2.9, centers[7][0] + 2.9, centers[7][0] + 2.8],
        [centers[1][1], centers[1][1], centers[7][1], centers[7][1]],
        color=C_SKIP,
        lw=1.0,
    )
    _arrow(
        ax,
        (centers[1][0] + 2.9, centers[1][1]),
        (centers[7][0] + 2.9, centers[7][1]),
        color=C_SKIP,
        lw=1.0,
    )
    ax.text(8.55, 14.55, "skip", fontsize=6.5, color=C_SKIP, rotation=90, va="center")
    ax.text(
        5,
        4.2,
        "Feature width $\\approx 32$ maps\nNo residual from input to output",
        ha="center",
        va="center",
        fontsize=7,
        color=C_NOTE,
    )
    _save(fig, path)


def render_restormer(path: Path, *, is_3d: bool) -> None:
    title = "Restormer3D" if is_3d else "Restormer-2D"
    down_lbl = "Down $2\\times$\n(strided Conv3d)" if is_3d else "Down $2\\times$\n(PixelUnshuffle)"
    up_lbl = "Up $2\\times$\n(ConvTranspose3d)" if is_3d else "Up $2\\times$\n(PixelShuffle)"
    mdta = "MDTA+GDFN (3D)" if is_3d else "MDTA+GDFN (2D)"
    slice_note = "" if is_3d else "\nSlice-wise 2D processing"

    fig, ax = plt.subplots(figsize=(6.8, 7.2), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(7, 13.5, title, ha="center", va="center", fontsize=12, fontweight="bold", color=C_EDGE)

    def box(x, y, w, h, text, fc=C_BLOCK, fs=7):
        _box(ax, x, y, w, h, text, facecolor=fc, fontsize=fs)

    box(3.0, 12.2, 3.2, 0.7, "Input  $K$ ch")
    box(3.0, 11.2, 3.2, 0.65, "Patch Embed")
    box(3.0, 10.1, 3.2, 0.95, f"Enc L1  $N=1$, $h=1$\n12 ch\n{mdta}", C_BLOCK, 6.5)
    box(3.0, 8.9, 3.0, 0.7, down_lbl, C_BLOCK, 6.5)
    box(3.0, 7.7, 3.2, 0.95, "Enc L2  $N=2$, $h=2$\n24 ch", C_BLOCK, 6.5)
    box(5.5, 7.7, 1.4, 0.5, "FiLM", C_FILM, 6.5)
    box(3.0, 6.5, 3.0, 0.7, down_lbl, C_BLOCK, 6.5)
    box(3.0, 5.2, 3.2, 0.95, "Latent  $N=2$\n48 ch", C_LATENT, 6.5)
    box(5.5, 5.2, 1.4, 0.5, "FiLM", C_FILM, 6.5)

    box(10.5, 6.5, 3.0, 0.7, up_lbl, C_DEC, 6.5)
    box(10.5, 7.7, 3.2, 0.95, "Dec L2  $N=2$\n$h=2$", C_DEC, 6.5)
    box(8.2, 7.7, 1.4, 0.5, "FiLM", C_FILM, 6.5)
    box(10.5, 8.9, 3.0, 0.7, up_lbl, C_DEC, 6.5)
    box(10.5, 10.1, 3.2, 0.95, "Dec L1  $N=1$\n$h=1$", C_DEC, 6.5)
    box(10.5, 11.2, 3.2, 0.65, "Refine  $N=2$", C_DEC, 7)
    box(10.5, 12.2, 3.4, 0.7, "Conv+PReLU\n$\\gamma\\cdot x+\\beta$", C_BLOCK, 6.5)
    box(10.5, 13.15, 3.0, 0.55, "Output  $1$ ch", C_BLOCK, 7.5)

    for y0, y1 in [(11.85, 11.55), (10.85, 10.6), (9.6, 9.25), (8.55, 8.2), (7.2, 6.85), (6.15, 5.7)]:
        _arrow(ax, (3.0, y0), (3.0, y1))
    _arrow(ax, (4.6, 5.2), (9.0, 6.5), connectionstyle="arc3,rad=-0.15")
    for y0, y1 in [(6.85, 7.2), (8.2, 8.55), (9.25, 9.6), (10.6, 10.85), (11.55, 11.85), (12.55, 12.85)]:
        _arrow(ax, (10.5, y0), (10.5, y1))
    _arrow(ax, (4.6, 10.1), (8.9, 10.1), color=C_SKIP, lw=1.0, connectionstyle="arc3,rad=-0.25")
    _arrow(ax, (4.6, 7.7), (8.9, 7.7), color=C_SKIP, lw=1.0, connectionstyle="arc3,rad=-0.25")
    ax.text(6.7, 10.35, "skip", fontsize=6, color=C_SKIP, ha="center")
    ax.text(6.7, 7.95, "skip", fontsize=6, color=C_SKIP, ha="center")
    ax.text(7.2, 5.85, "$[\\cos_x,\\cos_y,\\cos_z,b_{\\mathrm{norm}}]$", fontsize=6.5, color=C_NOTE, ha="center")
    ax.text(
        7,
        3.9,
        f"Widths $12\\rightarrow 24\\rightarrow 48$; blocks $(1,2,2)$; heads $(1,2,2)$"
        f"{slice_note}\nNo residual from input to output",
        ha="center",
        va="center",
        fontsize=7,
        color=C_NOTE,
    )
    ax.text(1.1, 10.1, "12 ch", fontsize=6.5, color=C_NOTE, ha="center")
    ax.text(1.1, 7.7, "24 ch", fontsize=6.5, color=C_NOTE, ha="center")
    ax.text(1.1, 5.2, "48 ch", fontsize=6.5, color=C_NOTE, ha="center")
    _save(fig, path)


def render_plain_cnn(path: Path) -> None:
    fig, ax = _setup((4.2, 7.0))
    ax.set_ylim(0, 14)
    ax.text(5, 13.3, "Plain-CNN-2D", ha="center", fontsize=11, fontweight="bold", color=C_EDGE)
    nodes = [
        (12.0, 0.8, "Input\n$K$ ch", C_BLOCK, 8),
        (10.5, 0.85, "Conv2d  $3{\\times}3$\n$K \\rightarrow 32$,  PReLU", C_BLOCK, 7.5),
        (8.9, 0.85, "Conv2d  $3{\\times}3$\n$32 \\rightarrow 32$,  PReLU", C_BLOCK, 7.5),
        (7.3, 0.85, "Conv2d  $3{\\times}3$\n$32 \\rightarrow 1$", C_BLOCK, 7.5),
        (5.8, 0.8, "Output\n$1$ ch", C_BLOCK, 8),
    ]
    _vchain(ax, nodes, x=4.0)
    ax.text(
        5,
        3.6,
        "No skip connections\nNo gating or recurrence\nNo attention\nSlice-wise 2D processing",
        ha="center",
        va="center",
        fontsize=7.5,
        color=C_NOTE,
        linespacing=1.35,
    )
    _save(fig, path)


def render_res_cnn(path: Path) -> None:
    fig, ax = _setup((4.4, 8.0))
    ax.set_ylim(0, 15)
    ax.text(5, 14.3, "Res-CNN-2D", ha="center", fontsize=11, fontweight="bold", color=C_EDGE)
    nodes = [
        (13.1, 0.8, "Input\n$K$ ch", C_BLOCK, 8),
        (11.7, 0.75, "Embed  Conv2d  $3{\\times}3$\n$K \\rightarrow \\mathrm{dim}\\ ({\\approx}24)$", C_BLOCK, 7),
        (9.7, 1.4, "Residual blocks $\\times 2$\nConv–GELU–Conv\n$+$ residual shortcut", C_PROC, 7.5),
        (7.8, 0.75, "Proj  Conv2d  $3{\\times}3$\n$\\mathrm{dim} \\rightarrow 1$", C_BLOCK, 7),
        (6.4, 0.8, "Output\n$1$ ch", C_BLOCK, 8),
    ]
    centers = _vchain(ax, nodes, x=4.2)
    _arrow(
        ax,
        (centers[2][0] + 2.95, centers[2][1] - 0.35),
        (centers[2][0] + 2.95, centers[2][1] + 0.35),
        color=C_SKIP,
        lw=1.1,
        connectionstyle="arc3,rad=0.6",
    )
    ax.text(8.6, 9.7, "residual", fontsize=6.5, color=C_SKIP, ha="left")
    ax.text(
        5,
        4.2,
        "No attention\nSlice-wise 2D processing",
        ha="center",
        va="center",
        fontsize=7.5,
        color=C_NOTE,
        linespacing=1.35,
    )
    _save(fig, path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    render_drcnet3d(OUT / "arch_drcnet3d")
    render_restormer(OUT / "arch_restormer3d", is_3d=True)
    render_restormer(OUT / "arch_restormer2d", is_3d=False)
    render_plain_cnn(OUT / "arch_plain_cnn2d")
    render_res_cnn(OUT / "arch_res_cnn2d")
    for name in (
        "arch_drcnet3d",
        "arch_restormer3d",
        "arch_restormer2d",
        "arch_plain_cnn2d",
        "arch_res_cnn2d",
    ):
        print((OUT / f"{name}.png").stat().st_size, name)


if __name__ == "__main__":
    main()
