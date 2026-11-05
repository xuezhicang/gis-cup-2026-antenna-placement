# Data

Download the datasets into this folder with:

```bash
bash data/download_data.sh
```

| File | Content |
|---|---|
| `GIS-cup-sample-dataset.geojson` | Sample dataset, 12,860 buildings |
| `GIS-cup-competition-dataset.geojson` | Competition dataset, 50,000 buildings |
| `competition-parameters.txt` | The competition's tau and k values |

All footprints are simple polygons without holes, in UTM zone 11N (EPSG:32611),
so coordinates are in metres.

The files come from the official evaluator repository,
<https://github.com/alowe/gis-cup-2026-evaluator>. They are derived from
Microsoft's Global ML Building Footprints and licensed under the
Community Data License Agreement – Permissive 2.0; see `NOTICE.md` (copied
unchanged from the evaluator repository) and `CDLA-Permissive-2.0.txt`.
