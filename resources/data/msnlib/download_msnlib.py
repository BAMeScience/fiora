#!/usr/bin/env python3
"""Download MSnLib archives or files.

Examples:
  python3 resources/data/msnlib/download_msnlib.py \
    --url "https://example.org/msnlib.mgf" \
    --output-dir resources/data/msnlib/raw

  python3 resources/data/msnlib/download_msnlib.py \
    --url "https://example.org/msnlib.zip" \
    --output-dir resources/data/msnlib/raw \
    --extract
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, urlretrieve


DEFAULT_URL = "https://zenodo.org/records/16984129"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "raw"


def _default_filename(url: str) -> str:
    name = Path(urlparse(url).path).name
    return name or "msnlib_download"


def _parse_zenodo_record_id(url: str) -> str | None:
    m = re.search(r"zenodo\.org/(?:records|record)/(\d+)", url)
    if not m:
        return None
    return m.group(1)


def _resolve_zenodo_record(
    url: str,
    *,
    filename_override: str | None,
    record_pattern: str | None,
    record_max_files: int,
) -> list[tuple[str, str]]:
    record_id = _parse_zenodo_record_id(url)
    if record_id is None or "/files/" in urlparse(url).path:
        resolved_url = url
        filename = filename_override or _default_filename(resolved_url)
        return [(resolved_url, filename)]

    api_url = f"https://zenodo.org/api/records/{record_id}"
    with urlopen(api_url) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    files = payload.get("files", [])
    if not files:
        raise RuntimeError(f"No downloadable files found in Zenodo record {record_id}")

    selected = []
    for file_item in files:
        key = str(file_item.get("key", ""))
        if not record_pattern or fnmatch.fnmatch(key, record_pattern):
            selected.append(file_item)

    if not selected:
        raise RuntimeError(
            f"No files in Zenodo record {record_id} match pattern {record_pattern!r}"
        )

    selected = sorted(selected, key=lambda x: str(x.get("key", "")))
    if record_max_files > 0:
        selected = selected[:record_max_files]

    if filename_override is not None and len(selected) != 1:
        raise RuntimeError(
            "--filename can only be used when exactly one file is selected"
        )

    resolved = []
    for file_item in selected:
        links = file_item.get("links", {})
        resolved_url = links.get("self")
        if not resolved_url:
            continue
        key = str(file_item.get("key") or _default_filename(resolved_url))
        filename = filename_override or key
        resolved.append((resolved_url, filename))

    if not resolved:
        raise RuntimeError(
            f"Could not resolve any download URLs for record {record_id}"
        )
    return resolved


def _extract_archive(path: Path, output_dir: Path) -> None:
    lower = path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(output_dir)
        return
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(output_dir)
        return
    if lower.endswith(".gz") and not lower.endswith(".tar.gz"):
        out_path = output_dir / path.with_suffix("").name
        with gzip.open(path, "rb") as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return
    raise ValueError(f"Unsupported archive format: {path}")


def _is_archive(path: Path) -> bool:
    lower = path.name.lower()
    return (
        lower.endswith(".zip")
        or lower.endswith(".tar.gz")
        or lower.endswith(".tgz")
        or (lower.endswith(".gz") and not lower.endswith(".tar.gz"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MSnLib files.")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=(
            "URL to download. Zenodo record URLs are supported and resolved to a file "
            "(default: MSnLib v7 Zenodo record)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store downloads/extracted files.",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Optional filename override for the downloaded file.",
    )
    parser.add_argument(
        "--record-pattern",
        default="*_ms2.mgf",
        help=(
            "Glob pattern for file keys when --url is a Zenodo record "
            "(default: *_ms2.mgf)."
        ),
    )
    parser.add_argument(
        "--record-max-files",
        type=int,
        default=0,
        help=(
            "Limit number of selected files from a Zenodo record. "
            "0 means no limit (default)."
        ),
    )
    parser.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract archives after download (default: true).",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_downloads = _resolve_zenodo_record(
        args.url,
        filename_override=args.filename,
        record_pattern=args.record_pattern,
        record_max_files=args.record_max_files,
    )
    print(f"Selected {len(resolved_downloads)} file(s) from {args.url}")

    for resolved_url, filename in resolved_downloads:
        dest = output_dir / filename
        print(f"Downloading {resolved_url} -> {dest}")
        urlretrieve(resolved_url, dest)

        if args.extract:
            if _is_archive(dest):
                extract_dir = output_dir / dest.stem
                extract_dir.mkdir(parents=True, exist_ok=True)
                _extract_archive(dest, extract_dir)
                print(f"Extracted to {extract_dir}")
            else:
                print(f"No extraction performed for {dest.name} (not an archive).")


if __name__ == "__main__":
    main()
