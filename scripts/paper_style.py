"""The paper's figure style and file output, shared by the paper_artifacts_* scripts.

Importing it sets Matplotlib's style. Text is set with LaTeX when it is installed, as
in the paper, and by Matplotlib otherwise.
"""

import io
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Global style. Figures are drawn at their printed size, so font sizes here are the
# sizes on the page when a figure is included at its intended width.
TEXT_WIDTH = 5.5  # inches
plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.6,
    "figure.constrained_layout.use": True,
    # Embed TrueType rather than Type 3 fonts.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Text set with LaTeX, as in the paper. The preamble mirrors neurips_2026.sty, which sets
# these two fonts and leaves math in Computer Modern.
LATEX_STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\renewcommand{\rmdefault}{ptm}\renewcommand{\sfdefault}{phv}",
}


def latex_is_installed():
    """Whether Matplotlib can set text with LATEX_STYLE, found by saving a small figure."""
    fig = plt.figure()
    try:
        with plt.rc_context(LATEX_STYLE):
            fig.text(0, 0, r"Text $\alpha$")
            for format in ("pdf", "png"):
                fig.savefig(io.BytesIO(), format=format)
        return True
    except (RuntimeError, OSError):
        return False
    finally:
        plt.close(fig)


USE_LATEX = latex_is_installed()
if USE_LATEX:
    plt.rcParams.update(LATEX_STYLE)
else:
    print("LaTeX not found: text is set by Matplotlib, so fonts differ from the paper.",
          file=sys.stderr)

# Matplotlib otherwise records itself and the time of saving in each file.
NO_METADATA = {
    ".pdf": {"Creator": None, "Producer": None, "CreationDate": None},
    ".png": {"Software": None},
}
PNG_DPI = 300


def save(fig, path, format="pdf"):
    """Write a figure to path.<format>, or a LaTeX table to path.tex; return the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(fig, str):  # A LaTeX table.
        path = path.with_suffix(".tex")
        path.write_text(fig)
        return path
    path = path.with_suffix(f".{format}")
    dpi = {"dpi": PNG_DPI} if format == "png" else {}
    fig.savefig(path, metadata=NO_METADATA[path.suffix], **dpi)
    plt.close(fig)
    return path
