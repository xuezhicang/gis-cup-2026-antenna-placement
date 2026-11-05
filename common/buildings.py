"""Convert building footprints to the flat arrays used by the C++ engines."""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon


def polygon_points_from_geometry(geom):
    if geom is None or geom.is_empty:
        return None

    if isinstance(geom, Polygon):
        coords = list(geom.exterior.coords)
    elif isinstance(geom, MultiPolygon):
        # The C++ tree stores one polygon per building. For a MultiPolygon,
        # use the geometry envelope so the tree bbox still covers all parts.
        coords = list(geom.envelope.exterior.coords)
    else:
        return None

    if len(coords) > 1 and coords[0] == coords[-1]:
        coords = coords[:-1]

    if len(coords) < 3:
        return None

    return [(float(x), float(y)) for x, y in coords]


def build_points_and_offsets(gdf):
    points = []
    offsets = [0]
    source_indices = []

    for source_idx, geom in enumerate(gdf.geometry):
        polygon = polygon_points_from_geometry(geom)
        if polygon is None:
            continue

        points.extend(polygon)
        offsets.append(len(points))
        source_indices.append(source_idx)

    if not points:
        raise ValueError("No valid Polygon or MultiPolygon geometries were found.")

    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(offsets, dtype=np.int32),
        np.asarray(source_indices, dtype=np.int32),
    )
