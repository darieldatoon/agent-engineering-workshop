"""Swap `> 📸 **<slug>** — ...` placeholders for `assets/screenshots/<slug>.png` once it exists.

Run: python3 tools/embed_screenshots.py   (prints what's still missing)
"""

import json
import os
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHOTS = ROOT / "assets" / "screenshots"
PLACEHOLDER = re.compile(r"^> 📸 \*\*([\w-]+)\*\* — (.*)$")


def replace_placeholders(text: str, rel: str) -> tuple[str, list[str]]:
    lines, out, missing, i = text.split("\n"), [], [], 0
    while i < len(lines):
        m = PLACEHOLDER.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        block = [m.group(2)]
        i += 1
        while i < len(lines) and lines[i].startswith(">"):
            block.append(lines[i].lstrip("> ").strip())
            i += 1
        slug = m.group(1)
        if not (SHOTS / f"{slug}.png").exists():
            out.append(f"> 📸 **{slug}** — " + block[0])
            out.extend(f"> {b}" for b in block[1:])
            missing.append(slug)
            continue
        alt = re.sub(r"[`*_]", "", " ".join(b for b in block if b)).replace('"', "'")
        out.append(f'<img src="{rel}/{slug}.png" alt="{alt}" width="900">')
    return "\n".join(out), missing


def main() -> None:
    missing_all = []
    for nb_path in sorted((ROOT / "notebooks").rglob("*.ipynb")):
        nb = json.loads(nb_path.read_text())
        rel = os.path.relpath(SHOTS, nb_path.parent)
        changed = False
        for cell in nb["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            src = "".join(cell["source"])
            new, missing = replace_placeholders(src, rel)
            missing_all += [f"{nb_path.relative_to(ROOT)}: {s}" for s in missing]
            if new != src:
                cell["source"] = new.splitlines(keepends=True)
                changed = True
        if changed:
            nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("\n".join(missing_all) or "all screenshots embedded")


if __name__ == "__main__":
    main()
