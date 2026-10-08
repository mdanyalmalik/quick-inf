"""Download the pinned SmolLM2 files used by the reference scripts and tests.

Files go to tools/reference/.cache/ (gitignored). Each file is checked against
the size and SHA-256 recorded in src/catalogue/models.ts, so the reference and
the app always use byte-identical artifacts.

Usage: tools/reference/.venv/bin/python -I tools/reference/fetch_model.py [--with-q4]
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "HuggingFaceTB/SmolLM2-360M-Instruct"
REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951"

# path -> (size in bytes, sha256). Keep in sync with src/catalogue/models.ts.
FILES = {
    "tokenizer.json": (2104556, "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"),
    "onnx/model_q4f16.onnx": (272737275, "cc63370efc2aca6d5307518b85162777132cc5b8d68eeb8154ea9b5fce09ad46"),
}
OPTIONAL = {
    "onnx/model_q4.onnx": (387943246, "a4ffda96e65beafc6f6cef0cbcf9fdc1cbdd79c230906bf3897d190547c7a596"),
}

CACHE = Path(__file__).resolve().parent / ".cache" / REPO.replace("/", "__") / REVISION


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(rel: str, size: int, digest: str) -> Path:
    dest = CACHE / rel
    if dest.exists() and dest.stat().st_size == size and sha256_of(dest) == digest:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{rel}"
    print(f"downloading {rel} ({size / 1e6:.1f} MB)")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as out:
        while chunk := r.read(1 << 20):
            out.write(chunk)
    actual_size, actual = tmp.stat().st_size, sha256_of(tmp)
    if actual_size != size or actual != digest:
        raise SystemExit(f"{rel}: expected {size} bytes sha256 {digest}, got {actual_size} bytes sha256 {actual}")
    tmp.rename(dest)
    return dest


def main() -> None:
    files = dict(FILES)
    if "--with-q4" in sys.argv:
        files.update(OPTIONAL)
    for rel, (size, digest) in files.items():
        print(f"ok {fetch(rel, size, digest)}")


if __name__ == "__main__":
    main()
