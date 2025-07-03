"""Assign transcripts to Cellpose ROIs and write updated Parquet file.

This helper assumes the transcripts contain global coordinates.  Depending on
when ``transcripts.parquet`` was generated the columns may be named either
``global_x``/``global_y`` or ``x_location``/``y_location`` (the format produced
for Baysor segmentation).
"""

from pathlib import Path
import pandas as pd
from shapely.geometry import Polygon, Point
from roifile import roiread
import rtree


def assign_cells(
    root_path: Path,
    transcripts_path: Path | None = None,
    roi_path: Path | None = None,
    output_path: Path | None = None,
) -> None:
    """Assign cell ids to decoded transcripts using Cellpose ROIs.

    Parameters
    ----------
    root_path : Path
        Path to the experiment directory containing ``qi2labdatastore``.
    transcripts_path : Path, optional
        Parquet file with decoded transcripts. Defaults to
        ``root_path/qi2labdatastore/all_tiles_filtered_decoded_features/transcripts.parquet``.
    roi_path : Path, optional
        Path to the Cellpose ROI ``.zip`` file. Defaults to
        ``root_path/qi2labdatastore/segmentation/cellpose/imagej_rois/global_coords_rois.zip``.
    output_path : Path, optional
        File to save the updated transcripts. Defaults to
        ``root_path/qi2labdatastore/all_tiles_filtered_decoded_features/transcripts_updated.parquet``.
    """

    datastore_path = root_path / "qi2labdatastore"

    if transcripts_path is None:
        transcripts_path = (
            datastore_path
            / "all_tiles_filtered_decoded_features"
            / "transcripts.parquet"
        )

    if roi_path is None:
        roi_path = (
            datastore_path
            / "segmentation"
            / "cellpose"
            / "imagej_rois"
            / "global_coords_rois.zip"
        )

    if output_path is None:
        output_path = (
            datastore_path
            / "all_tiles_filtered_decoded_features"
            / "transcripts_updated.parquet"
        )

    df = pd.read_parquet(transcripts_path)

    # Determine the column names for global coordinates.  The dataframe
    # produced by :class:`~merfish3danalysis.PixelDecoder` may store the
    # coordinates as ``global_x``/``global_y`` or, if prepared for Baysor,
    # ``x_location``/``y_location``.
    if {"global_y", "global_x"}.issubset(df.columns):
        y_col, x_col = "global_y", "global_x"
    elif {"y_location", "x_location"}.issubset(df.columns):
        y_col, x_col = "y_location", "x_location"
    else:
        raise KeyError(
            "Could not find global coordinate columns. Expected either"
            " {'global_y','global_x'} or {'y_location','x_location'}"
        )

    rois = roiread(roi_path)
    polygons = [Polygon(roi.subpixel_coordinates[:, ::-1]) for roi in rois]

    rtree_index = rtree.index.Index()
    for idx, poly in enumerate(polygons):
        rtree_index.insert(idx, poly.bounds)

    def check_point(row: pd.Series) -> int:
        pt = Point(row[y_col], row[x_col])
        candidates = list(rtree_index.intersection(pt.bounds))
        for cand in candidates:
            if polygons[cand].contains(pt):
                return cand + 1
        return 0

    df["cell_id"] = df.apply(check_point, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    root = Path("/path/to/experiment")
    assign_cells(root)
