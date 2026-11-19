"""Translate the Piper asset's display colors and Phong finish to a linear preview material."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def srgb_to_linear(color):
    color = np.asarray(color, dtype=np.float64)
    return np.where(color <= 0.04045, color / 12.92, ((color + 0.055) / 1.055) ** 2.4)


def read_materials(path):
    path = Path(path)
    materials = []
    for element in ET.parse(path).findall("./asset/material"):
        if "rgba" not in element.attrib:
            continue
        rgba = np.fromstring(element.attrib["rgba"], sep=" ")
        shininess = float(element.get("shininess", 0.5))
        # MJCF uses a normalized Phong exponent; match its lobe width in the roughness-based preview shader.
        roughness = float(np.sqrt(2 / (128 * shininess + 2)))
        materials.append(
            dict(
                name=element.get("name"),
                color_srgb=rgba[:3].tolist(),
                color_linear=srgb_to_linear(rgba[:3]).tolist(),
                roughness=roughness,
                specular=float(element.get("specular", 0.5)),
                shininess=shininess,
            )
        )
    return dict(
        asset=str(path.resolve()), asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), materials=materials
    )


def match_material(catalog, color):
    matches = [
        material for material in catalog["materials"] if np.allclose(material["color_srgb"], color, atol=1e-6, rtol=0)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one original Piper material for imported color {color}, got {len(matches)}")
    return matches[0]
