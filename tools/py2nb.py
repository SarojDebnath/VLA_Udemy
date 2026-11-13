"""Convert '# %%' percent-format scripts to .ipynb and optionally execute them.

usage: python tools/py2nb.py SRC.py OUT.ipynb [--run]
"""
import os, sys, re, nbformat
from nbclient import NotebookClient


def convert(src, kernel="vla"):
    text = open(src, encoding="utf-8").read()
    parts = re.split(r"^# %%(.*)$", text, flags=re.M)
    cells = []
    for i in range(1, len(parts), 2):
        tag, body = parts[i], parts[i + 1].strip("\n")
        if "[markdown]" in tag:
            md = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(md))
        elif body.strip():
            cells.append(nbformat.v4.new_code_cell(body))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": kernel, "display_name": f"Python ({kernel})", "language": "python"}
    return nb


def main():
    src, out = sys.argv[1], sys.argv[2]
    kernel = sys.argv[sys.argv.index("--kernel") + 1] if "--kernel" in sys.argv else "vla"
    nb = convert(src, kernel)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if "--run" in sys.argv:
        os.environ["VLA_HEADLESS"] = "1"
        NotebookClient(nb, timeout=14400, kernel_name=kernel,
                       resources={"metadata": {"path": os.path.dirname(os.path.abspath(out))}}).execute()
    nbformat.write(nb, out)
    print("ok", out)


if __name__ == "__main__":
    main()
