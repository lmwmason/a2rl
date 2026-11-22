"""Generate the 2026 gate and its printed artwork for the Elodin preview."""
from io import BytesIO
import json
from pathlib import Path
from struct import pack

from PIL import Image


def artwork_png():
    alpha = Path(__file__).resolve().parents[1] / "assets/season2-gate-ink.a8"
    ink = Image.frombytes("L", (256, 256), alpha.read_bytes())
    image = Image.new("RGBA", ink.size, (236, 230, 245))
    image.putalpha(ink)
    png = BytesIO()
    image.save(png, format="PNG")
    return png.getvalue()


def write_gate(path, half_depth):
    half_opening, half_outer = .75, 1.35
    positions = [[], [], []]

    def panel(material, axis, plane, u0, u1, v0, v1):
        def point(u, v):
            xyz = [u, v]
            xyz.insert(axis, plane)
            # The gate's plane is perpendicular to +X and its top is +Z.
            return xyz[2], xyz[0], xyz[1]

        a, b, c, d = point(u0, v0), point(u1, v0), point(u1, v1), point(u0, v1)
        positions[material].extend((a, b, c, a, c, d))

    a, b, d = half_opening, half_outer, half_depth
    for face in (-d, d):
        for low, high in [(-b, -a), (a, b)]:
            panel(0, 2, face, low, high, -b, b)
            panel(0, 2, face, -a, a, low, high)
        artwork = face + (.002 if face > 0. else -.002)
        panel(2, 2, artwork, -b, b, -b, b)
    for edge in (-b, -a, a, b):
        outer = abs(edge) == b
        end = b if outer else a
        panel(0, 0, edge, -end, end, -d, d)
        panel(0, 1, edge, -end, end, -d, d)
    for face in (-d - .001, d + .001):
        for low, high in [(-a - .04, -a), (a, a + .04)]:
            panel(1, 2, face, low, high, -a - .04, a + .04)
            panel(1, 2, face, -a, a, low, high)
        panel(1, 2, face, -b, -b + .025, -b, b)
        panel(1, 2, face, b - .025, b, -b, b)

    views, accessors, primitives = [], [], []
    binary = bytearray()
    for i, vertices in enumerate(positions):
        offset = len(binary)
        for vertex in vertices:
            binary.extend(pack("<3f", *vertex))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset, "target": 34962})
        axes = list(zip(*vertices))
        accessors.append({"bufferView": i, "componentType": 5126, "type": "VEC3", "count": len(vertices),
                          "min": [min(axis) for axis in axes], "max": [max(axis) for axis in axes]})
        primitives.append({"attributes": {"POSITION": i}, "material": i})
    # The artwork uses the same front/rear UV mapping as the Rust renderer.
    offset = len(binary)
    for depth, side, height in positions[2]:
        u = 0.5 + (side if depth > 0. else -side) / (2 * b)
        v = 0.5 - height / (2 * b)
        binary.extend(pack("<2f", u, v))
    views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset, "target": 34962})
    accessors.append({"bufferView": len(views) - 1, "componentType": 5126,
                      "type": "VEC2", "count": len(positions[2])})
    primitives[2]["attributes"]["TEXCOORD_0"] = len(accessors) - 1
    png = artwork_png()
    views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(png)})
    binary.extend(png)
    binary.extend(b"\0" * (-len(binary) % 4))
    metadata = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}], "meshes": [{"primitives": primitives}],
        "materials": [
            {"doubleSided": True, "pbrMetallicRoughness": {"baseColorFactor": [.18, .035, .53, 1],
                                                         "metallicFactor": 0, "roughnessFactor": .85}},
            {"doubleSided": True, "emissiveFactor": [.86, .70, 1.0],
             "pbrMetallicRoughness": {"baseColorFactor": [.85, .70, 1.0, 1], "metallicFactor": 0}},
            {"doubleSided": True, "alphaMode": "MASK", "alphaCutoff": .24,
             "extensions": {"KHR_materials_unlit": {}},
             "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0}},
        ],
        "extensionsUsed": ["KHR_materials_unlit"],
        "textures": [{"source": 0}],
        "images": [{"bufferView": len(views) - 1, "mimeType": "image/png"}],
        "buffers": [{"byteLength": len(binary)}], "bufferViews": views, "accessors": accessors,
    }
    document = json.dumps(metadata, separators=(",", ":")).encode()
    document += b" " * (-len(document) % 4)
    data = pack("<4sII", b"glTF", 2, 28 + len(document) + len(binary))
    data += pack("<I4s", len(document), b"JSON") + document
    data += pack("<I4s", len(binary), b"BIN\0") + binary
    path.write_bytes(data)
