"""Shared matplotlib styling for thesis figures.

Usage
-----
    from plot_style import apply_style, fig_size, save

    apply_style()                       # set rcParams once, before plotting

    fig, ax = plt.subplots(figsize=fig_size())          # full text width
    fig, ax = plt.subplots(figsize=fig_size(0.5))       # half-width column
    fig, ax = plt.subplots(figsize=fig_size(aspect=1.0))  # square

    ax.plot(...)
    save(fig, "figures/trajectory_xy")  # writes figures/trajectory_xy.pdf

`fig_size` returns a (width, height) tuple in inches sized relative to the
LaTeX text width, so figures land in the document at their native size and
share consistent font sizes with the body text.
"""

import os

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.transforms import Bbox


def _register_open_sans():
    """Register the vendored Open Sans faces so matplotlib can select them.

    Prefers the static RIBBI faces (clean normal/bold/italic mapping); falls
    back to anything in fonts/ (e.g. the variable fonts) if they're absent.
    Returns True if at least one 'Open Sans' face is available afterwards.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    static = os.path.join(base, 'fonts', 'static')
    core = ['OpenSans-Regular.ttf', 'OpenSans-Italic.ttf',
            'OpenSans-Bold.ttf', 'OpenSans-BoldItalic.ttf']
    added = False
    for f in core:
        p = os.path.join(static, f)
        if os.path.isfile(p):
            fm.fontManager.addfont(p)
            added = True
    if not added:
        fdir = os.path.join(base, 'fonts')
        if os.path.isdir(fdir):
            for f in os.listdir(fdir):
                if f.lower().endswith(('.ttf', '.otf')):
                    fm.fontManager.addfont(os.path.join(fdir, f))
    return any(f.name == 'Open Sans' for f in fm.fontManager.ttflist)


def apply_style():
    """Apply shared rcParams for consistent, publication-ready figures."""
    have_open_sans = _register_open_sans()
    if not have_open_sans:
        print('plot_style: Open Sans not found in fonts/ — falling back to DejaVu Sans.')
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Open Sans', 'DejaVu Sans'],
        # Open Sans has no math glyphs (e.g. lambda); keep math on a sans
        # math font that blends with it and covers Greek/sub/superscripts.
        'mathtext.fontset': 'dejavusans',
        'figure.constrained_layout.use': True,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def mark(fig, stem):
    """Tag `fig` with the output stem the save loop should use.

    Decouples saving from the figure title: set this at figure creation and the
    save loop writes `<stem>.pdf` regardless of what the suptitle says (or whether
    there is one). Returns `fig` for convenience.
    """
    fig._save_stem = stem
    return fig


def fig_size(width_frac=1.0, aspect=0.6, textwidth_in=5.5):
    """Return (width, height) in inches for a figure.

    Parameters
    ----------
    width_frac : float
        Fraction of the text width the figure should span (e.g. 0.5 for a
        half-width column).
    aspect : float
        Height / width ratio.
    textwidth_in : float
        LaTeX text width in inches (default 5.5).
    """
    w = textwidth_in * width_frac
    h = w * aspect
    return (w, h)


def save(fig, path_stem, tight=True):
    """Save `fig` to `path_stem` + ".pdf", creating parent dirs as needed.

    tight=True  -> crop to content (default; good for removing margins).
    tight=False -> save the full figure canvas with no cropping, so the PDF
                   width equals the authored figsize exactly. Use this for
                   figures that must scale identically in LaTeX (e.g. several
                   placed at the same \\includegraphics width).
    """
    path = path_stem + '.pdf'
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if tight:
        fig.savefig(path, bbox_inches='tight')
    else:
        w, h = fig.get_size_inches()
        fig.savefig(path, bbox_inches=Bbox.from_bounds(0, 0, w, h))
