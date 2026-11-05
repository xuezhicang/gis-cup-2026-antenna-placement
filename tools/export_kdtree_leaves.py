#!/usr/bin/env python3
"""Build the C++ building KD-tree and export its leaf boxes to a GeoPackage.

Debugging/visualization helper; not part of the main pipeline.

    python tools/export_kdtree_leaves.py --geojson data/GIS-cup-sample-dataset.geojson
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "build"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import time

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

import building_kdtree_cpp
from common.buildings import build_points_and_offsets


DEFAULT_GEOJSON = PROJECT_ROOT / "data" / "GIS-cup-sample-dataset.geojson"
DEFAULT_LEAF_GPKG = Path("leaf_nodes.gpkg")

LEAF_BOX_COLUMNS = [
    "node_id",
    "child_0",
    "child_1",
    "child_2",
    "child_3",
    "start",
    "end",
    "depth",
    "split_x",
    "split_y",
    "minx",
    "miny",
    "maxx",
    "maxy",
]

LEAF_BOX_INT_COLUMNS = [
    "node_id",
    "child_0",
    "child_1",
    "child_2",
    "child_3",
    "start",
    "end",
    "depth",
]


def leaf_boxes_to_gdf(leaf_boxes, crs):
    df = pd.DataFrame(leaf_boxes, columns=LEAF_BOX_COLUMNS)
    df[LEAF_BOX_INT_COLUMNS] = df[LEAF_BOX_INT_COLUMNS].astype(np.int64)
    df["building_count"] = df["end"] - df["start"]

    geometry = [
        box(row.minx, row.miny, row.maxx, row.maxy)
        for row in df.itertuples(index=False)
    ]
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)


def write_leaf_boxes_gpkg(leaf_boxes, crs, output_path, layer):
    leaf_gdf = leaf_boxes_to_gdf(leaf_boxes, crs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    leaf_gdf.to_file(output_path, layer=layer, driver="GPKG")
    return leaf_gdf


def main():
    parser = argparse.ArgumentParser(
        description="Build the C++ BuildingKDTree from a GeoJSON building file."
    )
    parser.add_argument(
        "--geojson",
        type=Path,
        default=DEFAULT_GEOJSON,
        help=f"Path to the building GeoJSON. Default: {DEFAULT_GEOJSON}",
    )
    parser.add_argument(
        "--max-leaf-size",
        type=int,
        default=64,
        help="Maximum number of buildings in one KD-tree leaf.",
    )
    parser.add_argument(
        "--leaf-gpkg",
        type=Path,
        default=DEFAULT_LEAF_GPKG,
        help=f"Path for exported leaf-node polygons. Default: {DEFAULT_LEAF_GPKG}",
    )
    parser.add_argument(
        "--leaf-layer",
        default="leaf_nodes",
        help="GeoPackage layer name for exported leaf-node polygons.",
    )
    parser.add_argument(
        "--leaf-bbox",
        choices=("modified", "original"),
        default="modified",
        help="Use modified expanded bbox or original cell bbox for leaf polygons.",
    )
    parser.add_argument(
        "--no-leaf-gpkg",
        action="store_true",
        help="Skip writing the leaf-node GeoPackage.",
    )
    args = parser.parse_args()

    start_time = time.time()
    gdf = gpd.read_file(args.geojson)
    print("geojson:", args.geojson)
    print("rows in file:", len(gdf))
    print("crs:", gdf.crs)
    print("read time (s):", round(time.time() - start_time, 3))

    convert_start = time.time()
    # Convert the GeoDataFrame geometries to points and offsets for the C++ KD-tree
    points, offsets, source_indices = build_points_and_offsets(gdf)
    print("valid buildings:", len(source_indices))
    print("points shape:", points.shape)
    print("offsets shape:", offsets.shape)
    print("convert time (s):", round(time.time() - convert_start, 3))

    tree_start = time.time()
    # Build the C++ KD-tree using the points and offsets
    tree = building_kdtree_cpp.BuildingKDTree(
        points,
        offsets,
        args.max_leaf_size,
    )
    print("build time (s):", round(time.time() - tree_start, 3))
    print("tree buildings:", tree.num_buildings)
    print("tree nodes:", tree.num_nodes)
    print("tree leaves:", tree.num_leaves)

    use_modified_bbox = args.leaf_bbox == "modified"
    leaf_boxes = tree.leaf_boxes(use_modified_bbox)
    print(f"leaf_boxes ({args.leaf_bbox}) shape:", leaf_boxes.shape)
    print("first leaf box row:", leaf_boxes[0].tolist() if len(leaf_boxes) else None)

    if not args.no_leaf_gpkg:
        export_start = time.time()
        leaf_gdf = write_leaf_boxes_gpkg(
            leaf_boxes,
            gdf.crs,
            args.leaf_gpkg,
            args.leaf_layer,
        )
        print("leaf gpkg:", args.leaf_gpkg)
        print("leaf layer:", args.leaf_layer)
        print("leaf features:", len(leaf_gdf))
        print("leaf export time (s):", round(time.time() - export_start, 3))

    sample_geom = gdf.geometry.iloc[int(source_indices[0])]
    sample_point = sample_geom.representative_point()
    query = tree.query_point(sample_point.x, sample_point.y)
    print("sample query point:", (sample_point.x, sample_point.y))
    print("covering leaf node ids:", query["node_ids"])
    print("candidate building indices in tree:", query["building_indices"])
    print(
        "candidate source row indices:",
        source_indices[np.asarray(query["building_indices"], dtype=np.int32)].tolist(),
    )


if __name__ == "__main__":
    main()
