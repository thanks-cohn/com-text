#!/usr/bin/env python3
"""
ocr_text_svg_overlay_filemonster_style.py

PDF / image directory / single image -> SVG files with line-by-line OCR text overlay.

This version deliberately uses the same OCR extraction pattern as FileMonster:

    result = ocr_engine.predict(str(image_path))

    for page_result in result:
        texts  = page_result.get("rec_texts", [])
        scores = page_result.get("rec_scores", [])
        boxes  = page_result.get("rec_boxes")
        polys  = page_result.get("rec_polys")

That matters because PaddleOCR v3 result objects are dict-like. Over-converting
them with __dict__, json wrappers, or boolean `or` chains can hide the real keys
or crash on NumPy arrays.

Install:
  pip install pymupdf pillow paddleocr paddlepaddle

Usage:
  python ocr_text_svg_overlay_filemonster_style.py chapter.pdf --out out --max-pages 3 --show-boxes
  python ocr_text_svg_overlay_filemonster_style.py page.png --out out --show-boxes
  python ocr_text_svg_overlay_filemonster_style.py ./pages --out out --recursive --show-boxes
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import mimetypes
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

# Must happen before paddle/paddleocr import.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

from PIL import Image, ImageOps

try:
    import fitz
except Exception:
    fitz = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


@dataclass
class TextLine:
    id: str
    page: int
    text: str
    bbox: list[float]
    confidence: Optional[float]
    source: str
    polygon: Optional[list[list[float]]] = None


@dataclass
class PageResult:
    page: int
    image_path: str
    width: int
    height: int
    lines: list[TextLine]


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return stem or "page"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def bbox_from_poly(poly: Any) -> Optional[list[float]]:
    if poly is None:
        return None

    # Paddle rec_boxes is usually [x0, y0, x1, y1], often a NumPy array.
    try:
        if len(poly) == 4 and all(isinstance(float(v), float) for v in poly):
            x0, y0, x1, y1 = map(float, poly)
            return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    except Exception:
        pass

    pts = []
    try:
        for p in poly:
            if len(p) >= 2:
                pts.append((float(p[0]), float(p[1])))
    except Exception:
        return None

    if not pts:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalize_polygon(poly: Any) -> Optional[list[list[float]]]:
    if poly is None:
        return None
    pts = []
    try:
        # [x0,y0,x1,y1] box
        if len(poly) == 4 and not hasattr(poly[0], "__len__"):
            x0, y0, x1, y1 = map(float, poly)
            return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    except Exception:
        pass

    try:
        for p in poly:
            if len(p) >= 2:
                pts.append([float(p[0]), float(p[1])])
    except Exception:
        return None

    return pts or None


def render_pdf_to_images(pdf_path: Path, rendered_dir: Path, dpi: int, first_page: int, max_pages: Optional[int]) -> list[Path]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Run: pip install pymupdf")

    ensure_dir(rendered_dir)
    doc = fitz.open(str(pdf_path))
    start = max(0, first_page - 1)
    end = len(doc) if max_pages is None else min(len(doc), start + max_pages)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    out_paths = []
    for page_index in range(start, end):
        page_no = page_index + 1
        pix = doc[page_index].get_pixmap(matrix=matrix, alpha=False)
        out = rendered_dir / f"{safe_stem(pdf_path.name)}_page_{page_no:04d}.png"
        pix.save(str(out))
        out_paths.append(out)

    doc.close()
    return out_paths


def collect_images(input_path: Path, out_dir: Path, dpi: int, first_page: int, max_pages: Optional[int], recursive: bool) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return render_pdf_to_images(input_path, out_dir / "_work" / "rendered_pages", dpi, first_page, max_pages)

    if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTS:
        return [input_path]

    if input_path.is_dir():
        pattern = "**/*" if recursive else "*"
        return sorted(p for p in input_path.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    raise ValueError(f"Unsupported input: {input_path}")


def make_ocr_engine(lang: str):
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        raise RuntimeError("PaddleOCR is not installed. Run: pip install paddleocr paddlepaddle") from exc

    try:
        return PaddleOCR(lang=lang, use_textline_orientation=True)
    except TypeError:
        return PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)


def run_ocr_filemonster_style(ocr_engine: Any, image_path: Path, page_no: int, min_confidence: float, debug: bool = False) -> list[TextLine]:
    result = ocr_engine.predict(str(image_path))
    lines: list[TextLine] = []

    if debug:
        print("DEBUG result type:", type(result), flush=True)
        try:
            print("DEBUG result len:", len(result), flush=True)
        except Exception:
            pass

    for page_result in result:
        if debug:
            print("DEBUG page_result type:", type(page_result), flush=True)
            try:
                print("DEBUG keys:", list(page_result.keys()), flush=True)
            except Exception as e:
                print("DEBUG no keys:", repr(e), flush=True)

        # This is the exact FileMonster access style.
        texts = page_result.get("rec_texts", [])
        scores = page_result.get("rec_scores", [])
        boxes = page_result.get("rec_boxes")
        polys = page_result.get("rec_polys")

        if debug:
            try:
                print("DEBUG rec_texts len:", len(texts), "sample:", list(texts[:5]), flush=True)
            except Exception as e:
                print("DEBUG texts inspect failed:", repr(e), flush=True)
            try:
                print("DEBUG rec_boxes type:", type(boxes), "len:", len(boxes) if boxes is not None else None, flush=True)
            except Exception as e:
                print("DEBUG boxes inspect failed:", repr(e), flush=True)
            try:
                print("DEBUG rec_polys type:", type(polys), "len:", len(polys) if polys is not None else None, flush=True)
            except Exception as e:
                print("DEBUG polys inspect failed:", repr(e), flush=True)

        for i, text in enumerate(texts):
            clean = str(text).strip()
            if not clean:
                continue

            try:
                conf = float(scores[i]) if i < len(scores) else 1.0
            except Exception:
                conf = 1.0

            if conf < min_confidence:
                continue

            bbox = None
            poly = None

            try:
                if boxes is not None and i < len(boxes):
                    bbox = bbox_from_poly(boxes[i])
                    poly = boxes[i]
            except Exception:
                bbox = None

            if bbox is None:
                try:
                    if polys is not None and i < len(polys):
                        bbox = bbox_from_poly(polys[i])
                        poly = polys[i]
                except Exception:
                    bbox = None

            if bbox is None:
                continue

            lines.append(TextLine(
                id=f"P{page_no}_L{len(lines)+1}",
                page=page_no,
                text=clean,
                bbox=bbox,
                confidence=conf,
                source="paddleocr_filemonster_style",
                polygon=normalize_polygon(poly),
            ))

    return sort_lines_reading_order(lines)


def sort_lines_reading_order(lines: list[TextLine]) -> list[TextLine]:
    if not lines:
        return lines

    heights = [max(1.0, ln.bbox[3] - ln.bbox[1]) for ln in lines]
    median_h = sorted(heights)[len(heights) // 2]
    row_bucket = max(8.0, median_h * 0.75)

    sorted_lines = sorted(lines, key=lambda ln: (round(ln.bbox[1] / row_bucket), ln.bbox[0]))
    for idx, ln in enumerate(sorted_lines, start=1):
        ln.id = f"P{ln.page}_L{idx}"
    return sorted_lines


def estimate_font_size(bbox: list[float]) -> float:
    return max(4.0, (bbox[3] - bbox[1]) * 0.82)


def write_json(page: PageResult, out_path: Path) -> None:
    data = {
        "page": page.page,
        "image_path": page.image_path,
        "width": page.width,
        "height": page.height,
        "lines": [asdict(ln) for ln in page.lines],
    }
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_svg(page: PageResult, out_path: Path, embed_image: bool, show_boxes: bool, hide_text: bool) -> None:
    img_path = Path(page.image_path)

    if embed_image:
        href = image_to_data_uri(img_path)
    else:
        href = os.path.relpath(img_path, out_path.parent)

    parts = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{page.width}" height="{page.height}" '
        f'viewBox="0 0 {page.width} {page.height}" data-source-image="{html.escape(str(img_path))}">'
    )
    parts.append('  <defs>')
    parts.append('    <style><![CDATA[')
    parts.append('      .text-line-object text { font-family: Arial, Helvetica, sans-serif; pointer-events: all; }')
    parts.append('      .ocr-box { fill: none; stroke: #0095ff; stroke-width: 1.25; }')
    parts.append('      .ocr-poly { fill: none; stroke: #ff9500; stroke-width: 1.0; }')
    parts.append('    ]]></style>')
    parts.append('  </defs>')
    parts.append(f'  <image class="page-image" x="0" y="0" width="{page.width}" height="{page.height}" href="{href}" />')
    parts.append(f'  <g id="ocr-text-lines" data-line-count="{len(page.lines)}">')

    for ln in page.lines:
        x0, y0, x1, y1 = ln.bbox
        fs = estimate_font_size(ln.bbox)
        baseline_y = y1 - max(1.0, fs * 0.12)
        txt = html.escape(ln.text)
        title = html.escape(json.dumps(asdict(ln), ensure_ascii=False))

        parts.append(
            f'    <g id="{html.escape(ln.id)}" class="text-line-object" '
            f'data-text="{html.escape(ln.text)}" data-confidence="{ln.confidence}">'
        )
        parts.append(f'      <title>{title}</title>')

        if show_boxes:
            parts.append(
                f'      <rect class="ocr-box" x="{x0:.2f}" y="{y0:.2f}" '
                f'width="{x1-x0:.2f}" height="{y1-y0:.2f}" opacity="0.25" />'
            )
            if ln.polygon:
                pts = " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in ln.polygon)
                parts.append(f'      <polyline class="ocr-poly" points="{pts}" opacity="0.25" />')

        visibility = "hidden" if hide_text else "visible"
        parts.append(
            f'      <text x="{x0:.2f}" y="{baseline_y:.2f}" font-size="{fs:.2f}" '
            f'fill="blue" opacity="0.85" visibility="{visibility}">{txt}</text>'
        )
        parts.append('    </g>')

    parts.append('  </g>')
    parts.append('</svg>')
    out_path.write_text("\n".join(parts), encoding="utf-8")


def crop_lines(image_path: Path, lines: list[TextLine], crops_dir: Path, pad: int = 2) -> None:
    ensure_dir(crops_dir)
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    for ln in lines:
        x0, y0, x1, y1 = ln.bbox
        left = max(0, int(math.floor(x0)) - pad)
        top = max(0, int(math.floor(y0)) - pad)
        right = min(w, int(math.ceil(x1)) + pad)
        bottom = min(h, int(math.ceil(y1)) + pad)
        if right > left and bottom > top:
            img.crop((left, top, right, bottom)).save(crops_dir / f"{ln.id}.png")


def process(input_path: Path, out_dir: Path, args: argparse.Namespace) -> list[PageResult]:
    ensure_dir(out_dir)
    images = collect_images(input_path, out_dir, args.dpi, args.first_page, args.max_pages, args.recursive)

    if not images:
        raise RuntimeError("No images/pages found.")

    print(f"Input: {input_path}")
    print(f"Pages/images: {len(images)}")
    print(f"Output: {out_dir}")
    print("")

    svg_dir = ensure_dir(out_dir / "svg")
    json_dir = ensure_dir(out_dir / "json")
    crops_dir = ensure_dir(out_dir / "crops") if args.save_crops else None

    ocr = make_ocr_engine(args.lang)
    results = []

    for idx, image_path in enumerate(images, start=1):
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im)
            width, height = im.size

        print(f"[OCR] page {idx}/{len(images)}: {image_path.name}", flush=True)
        lines = run_ocr_filemonster_style(
            ocr,
            image_path,
            page_no=idx,
            min_confidence=args.min_confidence,
            debug=args.debug_ocr,
        )

        page = PageResult(
            page=idx,
            image_path=str(image_path),
            width=width,
            height=height,
            lines=lines,
        )
        results.append(page)

        base = f"{idx:04d}_{safe_stem(image_path.name)}"
        write_json(page, json_dir / f"{base}.json")
        write_svg(page, svg_dir / f"{base}.svg", embed_image=not args.no_embed_image, show_boxes=args.show_boxes, hide_text=args.hide_text)

        if crops_dir is not None:
            crop_lines(image_path, lines, ensure_dir(crops_dir / base))

        print(f"      lines: {len(lines)}", flush=True)

    manifest = {
        "tool": "ocr_text_svg_overlay_filemonster_style.py",
        "input": str(input_path),
        "pages": len(results),
        "total_lines": sum(len(r.lines) for r in results),
        "svg_dir": str(svg_dir),
        "json_dir": str(json_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FileMonster-style PaddleOCR -> SVG text-line overlay.")
    p.add_argument("input", help="PDF, image, or image directory.")
    p.add_argument("--out", default="ocr_text_svg_out", help="Output directory.")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--first-page", type=int, default=1)
    p.add_argument("--max-pages", type=int, default=None)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--lang", default="en")
    p.add_argument("--min-confidence", type=float, default=0.20)
    p.add_argument("--show-boxes", action="store_true")
    p.add_argument("--hide-text", action="store_true")
    p.add_argument("--no-embed-image", action="store_true")
    p.add_argument("--save-crops", action="store_true")
    p.add_argument("--debug-ocr", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not input_path.exists():
        print(f"ERROR: input does not exist: {input_path}", file=sys.stderr)
        return 2

    try:
        results = process(input_path, out_dir, args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("")
    print("Done.")
    print(f"SVG:  {out_dir / 'svg'}")
    print(f"JSON: {out_dir / 'json'}")
    print(f"Lines: {sum(len(r.lines) for r in results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
