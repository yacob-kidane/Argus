#!/bin/bash
#SBATCH --job-name=prepare-data
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/sharefs/user13/autoresearch-scale/logs/prepare.out
#SBATCH --error=/mnt/sharefs/user13/autoresearch-scale/logs/prepare.err

echo "Node: $SLURMD_NODENAME"
echo "Python: $(which python3) $(python3 --version)"

export PATH=$HOME/.local/bin:$PATH

# Install all deps including torch
pip3 install --user torch requests pyarrow tiktoken --quiet
pip3 install --user rustbpe --quiet || echo "rustbpe install failed, trying alternative"

echo "Torch check: $(python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"

cd /mnt/sharefs/user13/karpathy-autoresearch
python3 prepare.py --num-shards 4

echo "Done"
ls -la ~/.cache/autoresearch/ 2>/dev/null || echo "cache dir not found"
