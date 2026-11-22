"""Generate the unbranded 2026 gate for the Elodin renderer."""
import json
from struct import pack


def write_gate(path):
    half_opening, half_outer, half_depth = .75, 1.35, .40
    positions = [[], []]

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
    metadata = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}], "meshes": [{"primitives": primitives}],
        "materials": [
            {"doubleSided": True, "pbrMetallicRoughness": {"baseColorFactor": [.18, .035, .53, 1],
                                                         "metallicFactor": 0, "roughnessFactor": .85}},
            {"doubleSided": True, "emissiveFactor": [.86, .70, 1.0],
             "pbrMetallicRoughness": {"baseColorFactor": [.85, .70, 1.0, 1], "metallicFactor": 0}},
        ],
        "buffers": [{"byteLength": len(binary)}], "bufferViews": views, "accessors": accessors,
    }
    document = json.dumps(metadata, separators=(",", ":")).encode()
    document += b" " * (-len(document) % 4)
    data = pack("<4sII", b"glTF", 2, 28 + len(document) + len(binary))
    data += pack("<I4s", len(document), b"JSON") + document
    data += pack("<I4s", len(binary), b"BIN\0") + binary
    path.write_bytes(data)
