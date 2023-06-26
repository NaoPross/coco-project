import warnings
import inspect
import logging
log = logging.getLogger(__name__)

import IPython as ipy
import numpy as np


video_width = 500


def show_source(fn):
    src = inspect.getsource(fn)
    display(ipy.display.Code(src, language="py"))


def run_and_show_video(simulator):
    if not simulator.videofile.resolve().exists():
        log.info("Video not found, running simulator ...")
        simulator.run()

    display(ipy.display.Video(simulator.videofile, width=video_width,
                              html_attributes="autoplay")) # loop


def delete_video(simulator):
    f = simulator.videofile.resolve()
    if f.exists():
        f.unlink()


def print_matrix(name: str, m: np.ndarray):
    s = f"$${name} ="
    s += r"\begin{bmatrix}"
    rows, cols = m.shape
    for r in range(rows):
        s += " & ".join(map(str, m[r, :]))
        s += r"\\"
    s += r"\end{bmatrix}$$"
    display(ipy.display.Latex(s))

