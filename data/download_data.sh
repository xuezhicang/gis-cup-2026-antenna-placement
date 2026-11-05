#!/usr/bin/env bash
# Download the GIS Cup 2026 sample and competition datasets into this folder.
# Source: the official evaluator repository, https://github.com/alowe/gis-cup-2026-evaluator
set -euo pipefail

cd "$(dirname "$0")"
BASE="https://raw.githubusercontent.com/alowe/gis-cup-2026-evaluator/HEAD/datasets"

for file in \
    GIS-cup-sample-dataset.geojson \
    GIS-cup-competition-dataset.geojson \
    competition-parameters.txt
do
    if [[ -s "$file" ]]; then
        echo "Already present: $file"
        continue
    fi
    echo "Downloading $file"
    curl -fL --retry 3 -o "$file.part" "$BASE/$file"
    mv "$file.part" "$file"
done

echo "Done. The data is licensed under CDLA-Permissive-2.0 (see NOTICE.md)."
