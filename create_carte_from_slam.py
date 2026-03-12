from __future__ import annotations

from pathlib import Path
import argparse
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, label


def build_carte_from_pgm(pgm_path: Path, output_path: Path) -> None:
    img = Image.open(pgm_path).convert("L")
    data = np.array(img, dtype=np.uint8)

    # nav2_map_server trinary convention (default)
    # - occupied: 0
    # - free: 254/255
    # - unknown: 205
    occupied = data <= 50
    free = data >= 250
    unknown = ~(occupied | free)

    rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)

    # FREE => white
    rgb[free] = np.array([255, 255, 255], dtype=np.uint8)
    # OCCUPIED => red (compatible with WALL_RED in path_finding_pipeline_flat.py)
    rgb[occupied] = np.array([255, 0, 0], dtype=np.uint8)
    # UNKNOWN => black (treated as MISC_OBJECT => obstacle)
    rgb[unknown] = np.array([0, 0, 0], dtype=np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)


def build_track_walls_from_pgm(
    pgm_path: Path,
    output_path: Path,
    *,
    min_wall_area: int = 120,
    wall_thickness_px: int = 2,
) -> None:
    img = Image.open(pgm_path).convert("L")
    data = np.array(img, dtype=np.uint8)

    occupied = data <= 50

    # Light cleanup to reduce tiny artifacts while preserving wall shapes
    occupied_clean = binary_closing(occupied, structure=np.ones((3, 3), dtype=bool))

    labels, num_labels = label(occupied_clean, structure=np.ones((3, 3), dtype=bool))
    components: list[tuple[int, np.ndarray]] = []
    for idx in range(1, num_labels + 1):
        comp = labels == idx
        area = int(np.count_nonzero(comp))
        if area >= min_wall_area:
            components.append((area, comp))

    components.sort(key=lambda x: x[0], reverse=True)

    rgb = np.full((data.shape[0], data.shape[1], 3), 255, dtype=np.uint8)

    # Largest wall component = exterior (red), second largest = interior (green)
    wall_colors = [
        np.array([0, 255, 0], dtype=np.uint8),
        np.array([255, 0, 0], dtype=np.uint8),
    ]

    selected = components[:2]
    for wall_idx, (_, comp) in enumerate(selected):
        boundary = comp & ~binary_erosion(comp, structure=np.ones((3, 3), dtype=bool), border_value=0)
        if wall_thickness_px > 1:
            boundary = binary_dilation(
                boundary,
                structure=np.ones((wall_thickness_px, wall_thickness_px), dtype=bool),
            )
        rgb[boundary] = wall_colors[wall_idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a SLAM .pgm map to a carte.png compatible with path_finding_pipeline_flat.py"
    )
    parser.add_argument(
        "--pgm",
        type=Path,
        required=True,
        help="Path to map .pgm file saved by nav2_map_server/map_saver_cli",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "carte.png",
        help="Output carte.png path (default: Software/path_finding/carte.png)",
    )
    parser.add_argument(
        "--also-super-mega-fusion",
        action="store_true",
        help="Also write the same carte.png into Software/super_mega_fusion/carte.png",
    )
    parser.add_argument(
        "--save-figure",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "super_mega_fusion" / "figure_finale.png",
        help="Save a final figure image at the end of the process (default: Software/super_mega_fusion/figure_finale.png)",
    )
    parser.add_argument(
        "--style",
        choices=["trinary", "track-walls"],
        default="trinary",
        help="Output style: trinary (legacy) or track-walls (white bg + outer red + inner green)",
    )
    parser.add_argument(
        "--min-wall-area",
        type=int,
        default=120,
        help="Minimum connected-component area (pixels) to keep as a wall in track-walls mode",
    )
    parser.add_argument(
        "--wall-thickness-px",
        type=int,
        default=2,
        help="Rendered wall thickness in pixels in track-walls mode",
    )
    args = parser.parse_args()

    if args.style == "track-walls":
        build_track_walls_from_pgm(
            args.pgm,
            args.out,
            min_wall_area=max(1, args.min_wall_area),
            wall_thickness_px=max(1, args.wall_thickness_px),
        )
    else:
        build_carte_from_pgm(args.pgm, args.out)
    print(f"[OK] Wrote {args.out}")

    if args.also_super_mega_fusion:
        target = Path(__file__).resolve().parents[1] / "super_mega_fusion" / "carte.png"
        if args.style == "track-walls":
            build_track_walls_from_pgm(
                args.pgm,
                target,
                min_wall_area=max(1, args.min_wall_area),
                wall_thickness_px=max(1, args.wall_thickness_px),
            )
        else:
            build_carte_from_pgm(args.pgm, target)
        print(f"[OK] Wrote {target}")

    args.save_figure.parent.mkdir(parents=True, exist_ok=True)
    out_resolved = args.out.resolve()
    fig_resolved = args.save_figure.resolve()
    if out_resolved != fig_resolved:
        shutil.copy2(args.out, args.save_figure)
        print(f"[OK] Saved final figure {args.save_figure}")
    else:
        print(f"[OK] Final figure already at {args.save_figure}")


if __name__ == "__main__":
    main()
