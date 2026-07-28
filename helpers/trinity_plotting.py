import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from cycler import cycler
import os


STYLE_ENV_VAR = "TRINITY_PLOT_STYLE"
STYLE_CHOICES = (
    "paper",
    "conference",
    "conference_dark",
    "conference_light",
    "dark_transparent",
    "light_transparent",
    "light",
    "dark",
)


_STYLE_ALIASES = {
    "publication": "paper",
    "pub": "paper",
    "journal": "paper",
    "conference": "conference_dark",
    "conf": "conference_dark",
    "conference_dark": "conference_dark",
    "conference-dark": "conference_dark",
    "dark_conference": "conference_dark",
    "dark-conference": "conference_dark",
    "dark_transparent": "conference_dark",
    "dark-transparent": "conference_dark",
    "transparent": "conference_dark",
    "conference_light": "conference_light",
    "conference-light": "conference_light",
    "light_conference": "conference_light",
    "light-conference": "conference_light",
    "light_transparent": "conference_light",
    "light-transparent": "conference_light",
}


def normalise_plot_style(style=None, *, default="paper", env_var=STYLE_ENV_VAR):
    """Return the canonical style name, with environment override support."""
    requested = os.environ.get(env_var) or os.environ.get("PLOT_STYLE") or style or default
    style_norm = str(requested).strip().lower()
    style_norm = _STYLE_ALIASES.get(style_norm, style_norm)
    valid = {"paper", "conference_dark", "conference_light", "light", "dark"}
    if style_norm not in valid:
        allowed = ", ".join(STYLE_CHOICES)
        raise ValueError(
            f"Unknown plot style '{requested}'. Use one of: {allowed}. "
            f"Set {STYLE_ENV_VAR}=paper|conference_dark|conference_light to override scripts."
        )
    return style_norm


def add_style_argument(parser, *, default=None):
    """Add the standard repository-wide plotting style flag to an ArgumentParser."""
    parser.add_argument(
        "--style",
        choices=STYLE_CHOICES,
        default=default,
        help=(
            "Plot style. 'paper' is light/opaque; 'conference' and "
            "'conference_dark' are dark transparent; 'conference_light' is "
            "light transparent. Can also be set with TRINITY_PLOT_STYLE."
        ),
    )
    return parser


def apply_plot_style(
    style=None,
    *,
    font="Times New Roman",
    base_fontsize=None,
    linewidth=None,
    n_colors=6,
    cmap_name="plasma",
    min_cycle=256,
):
    """Apply a paper/conference plotting style with sensible defaults."""
    style_norm = normalise_plot_style(style, default="paper")
    if style_norm == "paper":
        return set_paper_style(
            style="paper",
            font=font,
            base_fontsize=10 if base_fontsize is None else base_fontsize,
            linewidth=1.5 if linewidth is None else linewidth,
            n_colors=n_colors,
            cmap_name=cmap_name,
            min_cycle=min_cycle,
        )
    return set_plot_style(
        style=style_norm,
        font=font,
        base_fontsize=13 if base_fontsize is None else base_fontsize,
        linewidth=2.0 if linewidth is None else linewidth,
        n_colors=n_colors,
        cmap_name=cmap_name,
        min_cycle=min_cycle,
    )


def _colors_from_cmap(*, cmap, n):
    n = int(n)
    if n <= 0:
        return []
    if n == 1:
        return [cmap(0.5)]
    xs = np.linspace(0.15, 0.95, n, endpoint=True)
    return [cmap(float(x)) for x in xs]


def _distinct_hsv_colors(n, *, s=0.65, v=0.90, a=1.0, h0=0.0):
    n = int(n)
    if n <= 0:
        return []
    phi = 0.6180339887498949
    hs = (h0 + phi * np.arange(n, dtype=float)) % 1.0
    return [
        (*mcolors.hsv_to_rgb((float(h), float(s), float(v))), float(a))
        for h in hs
    ]


def _make_long_color_cycle(*, n_colors, cmap_name="plasma", min_cycle=256):
    n_colors = int(n_colors)
    cycle_len = int(max(min_cycle, max(1, n_colors)))
    cmap = plt.get_cmap(str(cmap_name))
    base_len = int(max(0, n_colors))
    base = _colors_from_cmap(cmap=cmap, n=base_len)

    extra_len = int(max(0, cycle_len - base_len))
    extra = _distinct_hsv_colors(extra_len, h0=0.15)

    return list(base) + list(extra)


def set_plot_style(
    style="light",
    font="Times New Roman",
    base_fontsize=15,
    linewidth=2.0,
    n_colors=6,
    cmap_name="plasma",
    min_cycle=256,
):
    style_norm = normalise_plot_style(style, default="light")
    colors = _make_long_color_cycle(n_colors=n_colors, cmap_name=cmap_name, min_cycle=min_cycle)

    # --- If dark mode: overwrite the first color with white ---
    if style_norm in {"dark", "conference_dark"}:
        colors[0] = (1.0, 1.0, 1.0, 1.0)   # RGBA white

    light = dict(
        figure_facecolor="#FFFFFF",
        axes_facecolor="#FFFFFF",
        axes_edgecolor="#262626",
        text_color="#262626",
        tick_color="#262626",
        grid_color="#CCCCCC",
    )
    dark = dict(
        figure_facecolor="#111217",
        axes_facecolor="#111217",
        axes_edgecolor="#EAEAEA",
        text_color="#EAEAEA",
        tick_color="#EAEAEA",
        grid_color="#555555",
    )
    paper = dict(
        figure_facecolor="#FFFFFF",
        axes_facecolor="#FFFFFF",
        axes_edgecolor="#000000",
        text_color="#000000",
        tick_color="#000000",
        grid_color="#AAAAAA",
    )

    if style_norm in {"light"}:
        theme = light
    elif style_norm == "conference_light":
        theme = light.copy()
        theme["figure_facecolor"] = "none"
        theme["axes_facecolor"] = "none"
    elif style_norm in {"dark", "conference_dark"}:
        theme = dark.copy()
        if style_norm == "conference_dark":
            theme["figure_facecolor"] = "none"
            theme["axes_facecolor"] = "none"
    elif style_norm == "paper":
        theme = paper
    else:
        raise ValueError(
            "Unknown style. Use one of: 'paper', 'conference', 'conference_dark', 'conference_light', 'dark_transparent', 'light_transparent', 'light', 'dark'."
        )

    is_transparent = theme["figure_facecolor"] == "none"

    plt.rcParams.update({
        # Figure & font
        "figure.dpi": 150,
        "figure.autolayout": True,
        "font.family": "serif",
        # Fallback chain: honour the requested font first (Times New Roman on
        # macOS / MikTeX / TeXLive Mac builds), then STIX Two Text and STIX
        # General (Times-clone shipped with matplotlib, present in every
        # matplotlib install including sandbox Linux), TeX Gyre Termes (Times
        # metrics, ships with TeX Live), Liberation Serif (Times metrics,
        # ships with most Linux distros), and finally DejaVu Serif as the
        # absolute fallback. This keeps embedded PDF fonts consistent across
        # render environments — the paper set always looks Times-like.
        "font.serif": [font, "STIX Two Text", "STIXGeneral", "TeX Gyre Termes",
                        "Liberation Serif", "DejaVu Serif"],
        "font.size": base_fontsize,

        "mathtext.fontset": "cm",

        # Axes & text colors
        "axes.facecolor": theme["axes_facecolor"],
        "figure.facecolor": theme["figure_facecolor"],
        "axes.edgecolor": theme["axes_edgecolor"],
        "axes.labelcolor": theme["text_color"],
        "axes.titlecolor": theme["text_color"],
        "text.color": theme["text_color"],
        "xtick.color": theme["tick_color"],
        "ytick.color": theme["tick_color"],
        "grid.color": theme["grid_color"],

        # Sizes
        "axes.titlesize": base_fontsize + 2,
        "axes.labelsize": base_fontsize + 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,

        # Lines & cycle
        "axes.prop_cycle": cycler(color=colors),
        "lines.linewidth": linewidth,
        "lines.markersize": 5,

        "patch.edgecolor": theme["text_color"],
        "patch.facecolor": theme["text_color"],

        # Grid
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.alpha": 0.3,

        # Legend
        "legend.frameon": False,
        "legend.fontsize": base_fontsize - 1,
        "legend.labelcolor": theme["text_color"],

        # Colormap
        "image.cmap": str(cmap_name),

        # Save
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": theme["figure_facecolor"],
        "savefig.edgecolor": theme["figure_facecolor"],
        "savefig.transparent": bool(is_transparent),
    })

    plt.set_cmap(str(cmap_name))


def get_cmap_colors(*, cmap_name="plasma", n=6, start=0.15, end=0.95):
    n = int(n)
    if n <= 0:
        return []
    cmap = plt.get_cmap(str(cmap_name))
    if n == 1:
        return [cmap(float(0.5 * (start + end)))]
    xs = np.linspace(float(start), float(end), n, endpoint=True)
    return [cmap(float(x)) for x in xs]


def plasma_colors(*, n=6, start=0.15, end=0.95):
    return get_cmap_colors(cmap_name="plasma", n=n, start=start, end=end)


def plasma_color(x):
    return plt.get_cmap("plasma")(float(x))


def save_figure(fig, path_no_ext, *, dpi=200):
    fig.savefig(
        f"{path_no_ext}.png",
        dpi=dpi,
        transparent=bool(plt.rcParams.get("savefig.transparent", False)),
        facecolor=plt.rcParams.get("savefig.facecolor", "auto"),
        edgecolor=plt.rcParams.get("savefig.edgecolor", "auto"),
    )


def current_style_is_transparent():
    return bool(plt.rcParams.get("savefig.transparent", False))


def current_savefig_kwargs(**kwargs):
    """Return savefig kwargs consistent with the active Trinity style."""
    out = {
        "transparent": current_style_is_transparent(),
        "facecolor": plt.rcParams.get("savefig.facecolor", "auto"),
        "edgecolor": plt.rcParams.get("savefig.edgecolor", "auto"),
    }
    out.update(kwargs)
    return out


def set_paper_style(
    style=None,
    font="Times New Roman",
    base_fontsize=10,
    linewidth=1.5,
    n_colors=6,
    cmap_name="plasma",
    min_cycle=256,
    figure_width_inches=3.5,
):
    """
    Set publication-ready plot style with appropriate font sizes for journal figures.
    
    Parameters
    ----------
    font : str
        Font family (default: "Times New Roman")
    base_fontsize : float
        Base font size in points. For single-column figures (3.5"), use 8-10pt.
        For double-column figures (7"), use 9-11pt. (default: 10)
    linewidth : float
        Default line width (default: 1.5)
    n_colors : int
        Number of colors in the color cycle (default: 6)
    cmap_name : str
        Colormap name (default: "plasma")
    min_cycle : int
        Minimum color cycle length (default: 256)
    figure_width_inches : float
        Typical figure width for sizing reference. Single-column: 3.5", double-column: 7" (default: 3.5)
    
    Notes
    -----
    Font size guidelines for publication (assuming 3.5" single-column width):
    - Axis labels: 10-11 pt
    - Tick labels: 8-9 pt
    - Legend: 8-9 pt
    - Title (if used): 11-12 pt
    
    Colors are optimized for white backgrounds and print reproduction.
    """
    try:
        style_norm = normalise_plot_style(style, default="paper")
    except ValueError:
        # Backwards compatibility: before style selection was added, the first
        # positional argument to set_paper_style was the font name.
        if style is None:
            raise
        font = str(style)
        style_norm = normalise_plot_style(None, default="paper")
    if style_norm != "paper":
        set_plot_style(
            style=style_norm,
            font=font,
            base_fontsize=base_fontsize,
            linewidth=linewidth,
            n_colors=n_colors,
            cmap_name=cmap_name,
            min_cycle=min_cycle,
        )
        return

    set_plot_style(
        style=style_norm,
        font=font,
        base_fontsize=base_fontsize,
        linewidth=linewidth,
        n_colors=n_colors,
        cmap_name=cmap_name,
        min_cycle=min_cycle,
    )
    
    plt.rcParams.update({
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "savefig.dpi": 300,
    })


def annotate_auto_arrow(
    *,
    ax,
    text,
    xy,
    xytext,
    color=None,
    arrow_color=None,
    **kwargs,
):
    if color is None:
        color = plt.rcParams.get("text.color", "k")
    if arrow_color is None:
        arrow_color = color
    arrowprops = dict(kwargs.pop("arrowprops", {}) or {})
    arrowprops.setdefault("arrowstyle", "->")
    arrowprops.setdefault("color", arrow_color)
    arrowprops.setdefault("lw", plt.rcParams.get("lines.linewidth", 1.5))
    return ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        color=color,
        arrowprops=arrowprops,
        **kwargs,
    )
