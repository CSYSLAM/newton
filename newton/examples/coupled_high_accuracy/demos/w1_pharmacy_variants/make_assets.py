"""Generic, distinct carton print designs and the matching physical mesh archives."""

from pathlib import Path

import numpy as np
from catalog import CARTONS
from PIL import Image, ImageDraw, ImageFont
from variant_geometry import make_carton

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[1] / "assets/w1/pharmacy"
FONT = HERE.parents[1] / "assets/w1/fonts/NotoSansCJKsc-Regular.otf"


def main():
    folder = ASSETS
    folder.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(CARTONS):
        np.savez_compressed(folder / f"{case.key}_physics.npz", **make_carton(case))
        for panel in range(6):
            image = Image.new("RGB", (768, 1024), (246, 246, 240))
            d = ImageDraw.Draw(image)
            font = lambda n: ImageFont.truetype(str(FONT), n)
            ink = case.color
            gray = (60, 70, 69)
            d.rectangle((0, 0, 768, 86), fill=ink)
            d.text((44, 15), "青禾 QINGHE", font=font(38), fill="white")
            d.text((54, 155), case.product, font=font(124), fill=ink)
            d.text((60, 330), case.key.upper(), font=font(44), fill=gray)
            d.line((55, 424, 715, 424), fill=ink, width=3)
            contents = ("铝塑包装  20片", "胶囊包装  24粒", "独立袋装  12袋", "独立包装  8贴")[index]
            d.text((60, 475), contents, font=font(38), fill=gray)
            d.text((60, 565), "包装样品", font=font(30), fill=(125, 130, 125))
            # Fine rules make actual surface buckling visible in the printed graphic.
            for y in (655, 681, 707):
                d.line((60, y, 710, y), fill=(180, 185, 180), width=2)
            d.rectangle((0, 748, 768, 830), fill=ink)
            d.text((58, 762), "QINGHE · PHARMACY", font=font(34), fill="white")
            rng = np.random.default_rng(41 + index)
            x = 60
            for width in rng.integers(2, 7, 74):
                d.rectangle((x, 868, x + int(width), 958), fill=gray)
                x += int(width) + 3
            d.text((58, 976), f"6 921000 03852{index}", font=font(21), fill=gray)
            if panel >= 4:
                d.line((12, 112, 756, 112), fill=(130, 140, 132), width=2)
                d.text((54, 115), f"LOT 261030 / QH-{index + 1:02d}", font=font(25), fill=gray)
            image.save(folder / f"{case.key}_panel_{panel}.png")


if __name__ == "__main__":
    main()
