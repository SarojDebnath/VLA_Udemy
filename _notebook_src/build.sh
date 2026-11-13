#!/bin/bash
# Rebuild notebooks from their percent-script sources and execute them.
# usage: _notebook_src/build.sh SECTION_SRC_DIR REPO_SECTION_DIR [names...]
#   e.g. _notebook_src/build.sh _notebook_src/S04 S04_Classical_Control 42c_project_lqr_balance
# Uses the python in $VLA_PY, else the active environment's python.
V="${VLA_PY:-python}"
SRCD=$1; OUTD=$2; shift 2
cd "$(dirname "$0")/.."
for f in ${@:-$(ls $SRCD/*.py | xargs -n1 basename | sed 's/\.py$//')}; do
  $V tools/py2nb.py "$SRCD/$f.py" "$OUTD/$f.ipynb" --run 2>&1 | grep -E "^ok|Error|error|Traceback|Exception" | grep -v "^ *warnings" | head -8
done
