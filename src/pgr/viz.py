from __future__ import annotations
from typing import List, Tuple, Optional, Sequence, Union
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# ... keep your existing helpers (_ensure_uint8_rgb, _normalize01, etc.)

def grid_panel(paths_or_images: Sequence[Union[str, Path, np.ndarray, Image.Image]],
                rows: int = 3,
                cols: int = 3,
                title: Optional[str] = None,
                pad: int = 4) -> Image.Image:
    """Build a rows×cols grid from paths or images."""
    ims: List[Image.Image] = []
    for x in paths_or_images[:rows * cols]:
        if isinstance(x, Image.Image):
            im = x
        elif isinstance(x, (str, Path)):
            im = Image.open(x).convert("RGB")
        elif isinstance(x, np.ndarray):
            im = Image.fromarray(_ensure_uint8_rgb(x))
        else:
            raise TypeError(f"unsupported type: {type(x)}")
        ims.append(im)

    # make uniform size
    w, h = ims[0].size
    ims = [im.resize((w, h), Image.BILINEAR) if im.size != (w, h) else im for im in ims]
    panel = grid(ims, rows=rows, cols=cols, pad=pad)

    if title:
        # simple header strip
        tw, th = panel.size[0], 24
        header = Image.new("RGB", (tw, th), (20, 20, 20))
        out = Image.new("RGB", (tw, th + panel.size[1]), (0, 0, 0))
        out.paste(header, (0, 0))
        out.paste(panel, (0, th))
        d = ImageDraw.Draw(out)
        d.text((8, 4), title, fill=(240, 240, 240))
        return out
    return panel
