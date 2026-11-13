"""Remove repetitive simulator warnings (MJX/Warp solver-iteration prints) from a notebook's stream outputs."""
import re, sys, nbformat
NOISE = re.compile(r"solver iterations|warn_overflow|OverflowType|ITERATIONS|= 0 for all|^\s*\d+\s*$|Module .* load on device|^ *ll\)$|^ *ons beyond")
def clean(path):
    nb = nbformat.read(path, 4); removed = 0
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                lines = o["text"].splitlines(keepends=True)
                keep = [l for l in lines if not NOISE.search(l)]
                removed += len(lines) - len(keep); o["text"] = "".join(keep)
    nbformat.write(nb, path); print(f"{path}: removed {removed} noise lines")
if __name__ == "__main__":
    for p in sys.argv[1:]:
        clean(p)
