"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

MODULE_ROOT = "src"
MODULE_PREFIX = ""

nav = mkdocs_gen_files.Nav()

for path in sorted(Path(MODULE_ROOT).rglob("*.py")):
    module_path = path.relative_to(MODULE_ROOT).with_suffix("")
    doc_path = path.relative_to(MODULE_ROOT).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue
    elif parts[-1].startswith("_"):
        continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print(f"::: {MODULE_PREFIX}{identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
