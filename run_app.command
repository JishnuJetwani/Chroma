#!/bin/zsh
set -euo pipefail
cd -- "${0:A:h}"

if [[ ! -x .venv/bin/python ]]; then
  print 'Set up the environment first: uv sync --extra app --extra data --extra raw'
  exit 1
fi

photograder_checkpoint=''
for candidate in checkpoints/jpeg.pt runs/full-7030-20261012/joint/model.pt \
  runs/raw-pilot-20261030/shared/model.pt runs/multistyle_pilot/best.pt runs/local_pilot/best.pt; do
  if [[ -f "$candidate" ]]; then
    photograder_checkpoint="$candidate"
    break
  fi
done
if [[ -z "$photograder_checkpoint" ]]; then
  print 'Place trained weights at checkpoints/jpeg.pt, or follow the training guide in README.md.'
  exit 1
fi

print 'Selective Grade runs locally at http://127.0.0.1:7860'
exec .venv/bin/python -m photograder app --checkpoint "$photograder_checkpoint"
