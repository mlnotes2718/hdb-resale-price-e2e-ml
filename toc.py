# generate_toc.py
import nbformat, re, sys, pathlib
nb = nbformat.read(sys.argv[1], as_version=4)
def slug(txt):
    txt = re.sub(r'[^\w\s-]', '', txt).strip().lower()
    return re.sub(r'\s+', '-', txt)
lines = ["## Table of Contents"]
for cell in nb.cells:
    if cell.cell_type == "markdown":
        for m in re.finditer(r'^(#+)\s+(.*)', cell.source, re.M):
            lvl, txt = len(m.group(1)), m.group(2).strip()
            lines.append(f'{"  "*(lvl-1)}- [{txt}](#{slug(txt)})')
print("\n".join(lines))