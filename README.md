# HOTC 2026 2nd Place Solution

**HSI + Amodal Completion + Dynamic/Static Scene Detection + RTS Smoothing**

Open-source inference code from **Alpha AI** for the HOTC 2026 2nd-place solution.

- Final rank: **2nd place**
- Success AUC: **68.0093%**
- DP@20: **87.7703%**
- Reproducible Val75 pipeline: **0.69768**

## Method

1. **SAM3 tracking** — official SAM3 with a project-authored DAM/DRM adapter
   provides the base visible-object tracker.
2. **HSI verification** — spectral consistency detects appearance changes and
   helps distinguish dynamic and static scenes.
3. **Dynamic/static scene detection** — scene-aware gates control tracker
   updates and reset persistent one-sided box expansion.
4. **Amodal completion** — a temporal amodal head estimates the full object box
   under occlusion.
5. **RTS smoothing** — Rauch–Tung–Striebel smoothing fills empty-mask frames.

## Setup

Requirements: Linux, Python 3.10, Git, an NVIDIA CUDA GPU, and enough
disk space for the 3.45 GB SAM3 checkpoint.

```bash
git clone https://github.com/RyogaYuzawa/hotc2026-hyperdam.git
cd hotc2026-hyperdam

bash scripts/setup.sh

# Accept the SAM3 license at https://huggingface.co/facebook/sam3 first.
.venv/bin/hf auth login

bash scripts/download_weights.sh
python3 scripts/verify_artifacts.py --profile runtime
```

The setup script downloads a pinned revision of the official Meta SAM3 source.

## Weights

`scripts/download_weights.sh` downloads both checkpoints from Hugging Face:

- `weights/sam3.pt` — official
  [`facebook/sam3`](https://huggingface.co/facebook/sam3)
- `weights/amodal-v10.pt` —
  [`ryo818/HOTC2026-Amodal`](https://huggingface.co/ryo818/HOTC2026-Amodal)

The script uses fixed revisions and verifies both files. SAM3 requires accepting
Meta's license and authenticating with Hugging Face first.

Inference code is under `hotc/`; downloaded checkpoints are stored in `weights/`.

## Dataset

Download the prepared modal data from
[HOTC2026-Modal](https://huggingface.co/datasets/ryo818/HOTC2026-Modal).
Review and accept the official
[HOTC 2026 competition rules](https://www.kaggle.com/competitions/hyperspectral-object-tracking-challenge-2026)
before use, then arrange the files as follows:

```text
data/HOTC2026/
  sample_submisson.csv
  validation/
    HSI-NIR-FalseColor/<sequence>/0001.jpg
    HSI-NIR/<sequence>/0001.png
    HSI-RedNIR-FalseColor/<sequence>/0001.jpg
    HSI-RedNIR/<sequence>/0001.png
    HSI-VIS-FalseColor/<sequence>/0001.jpg
    HSI-VIS/<sequence>/0001.png
```

Each false-color sequence must include `init_rect.txt` containing
`x,y,width,height`. False-color and raw-HSI frames must have matching numeric
filenames.

The dataset, credentials, and model weights are not stored in Git.

## Inference

One GPU:

```bash
bash scripts/run_sample.sh data/HOTC2026 \
  --sample data/HOTC2026/sample_submisson.csv
```

Multiple GPUs:

```bash
HOTC_DEVICES=cuda:0,cuda:1,cuda:2,cuda:3 \
  bash scripts/run_sample.sh data/HOTC2026 \
  --sample data/HOTC2026/sample_submisson.csv
```

The output is written to `result/submission.csv`.

## Citation

```bibtex
@inproceedings{yuzawa2026hyperdam,
  title     = {HyperDAM: Hyperspectral Distractor-Aware Memory with Amodal Expansion
               for SAM 3 Tracking},
  author    = {Yuzawa, Ryoga and Takagi, Tasuku},
  booktitle = {2026 IEEE Workshop on Hyperspectral Image and Signal Processing:
               Evolution in Remote Sensing (WHISPERS)},
  year      = {2026}
}
```

## License

Project-authored code is licensed under [Apache-2.0](LICENSE), except the SAM3
integration identified in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
SAM3 integration code, source, and weights are subject to Meta's
[SAM License](THIRD_PARTY_LICENSES/SAM_LICENSE). Downloaded packages,
checkpoints, and datasets retain their own terms.
