#!/usr/bin/env python3
"""Build ghost-decode.zip, the uploadable claude.ai skill archive.

Run from anywhere:  python claude-ai-skill/build-zip.py

Produces claude-ai-skill/ghost-decode.zip containing only the skill folder
(ghost-decode/SKILL.md — the decoder is embedded in it) with forward-slash paths,
so it uploads cleanly to claude.ai. The zip is intentionally NOT committed to the
repo: a zip inside the repo becomes a nested zip inside the plugin package
(marketplace source "./"), which claude.ai refuses to install.
"""
import zipfile
import pathlib

here = pathlib.Path(__file__).resolve().parent
src = here / "ghost-decode"
out = here / "ghost-decode.zip"

if not (src / "SKILL.md").is_file():
    raise SystemExit(f"error: {src/'SKILL.md'} not found")

if out.exists():
    out.unlink()

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(src.rglob("*")):
        if f.is_file():
            z.write(f, "ghost-decode/" + f.relative_to(src).as_posix())

print(f"Wrote {out}")
with zipfile.ZipFile(out) as z:
    for name in z.namelist():
        print("  " + name)
print("\nUpload this single file in claude.ai -> Settings -> Capabilities -> "
      "Skills -> Upload skill. Do NOT upload a zip of the whole repo/folder.")
