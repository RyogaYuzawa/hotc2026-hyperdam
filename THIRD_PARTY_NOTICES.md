# Third-party notices

The root Apache-2.0 license covers project-authored files in this repository
except `hotc/sam3_tracker.py`, which is distributed under the SAM License due
to its close integration with SAM 3 internals. The root license does not
relicense downloaded source, Python packages, model checkpoints, or datasets.
Those materials are not tracked by Git and retain their respective terms.

## Meta SAM 3

`scripts/setup.sh` fetches the official SAM 3 source directly from
<https://github.com/facebookresearch/sam3> at commit
`20dba30a35a497606b06cf241f5b5605ea10e77e`.
`scripts/download_weights.sh` separately downloads the official SAM 3
checkpoint from <https://huggingface.co/facebook/sam3> at a fixed revision.

The source and checkpoint remain subject to Meta's SAM License. A verbatim
copy is provided at `THIRD_PARTY_LICENSES/SAM_LICENSE`. Review it before use.
Among other conditions, it requires a copy of the agreement with redistributed
SAM materials, acknowledgment in research publications, and compliance with
its use and trade-control restrictions.

The project-local streaming tracker and DAM/DRM memory adapter are in
`hotc/sam3_tracker.py`. It carries an explicit SAM License notice. No
SAM3-TrackBench or original DAM4SAM source is fetched, imported, or bundled.

## Amodal checkpoint

`scripts/download_weights.sh` downloads `amodal-v10.pt` from
<https://huggingface.co/ryo818/HOTC2026-Amodal>. Its model card declares
Apache-2.0. The checkpoint is not tracked by this Git repository. SAM 3 is not
included in that model repository and remains under the separate SAM License.

## Python packages

`scripts/setup.sh` installs Python packages into the user's local virtual
environment; this Git repository does not redistribute their source or wheels.
Each package retains its own license. A pip-installed CUDA build of PyTorch
also installs NVIDIA CUDA libraries under NVIDIA's proprietary license terms.
NumPy and SciPy binary wheels can contain GCC runtime components under the GPL
with the GCC Runtime Library Exception and libquadmath under LGPL-2.1-or-later.
Those bundled components do not change the license of this repository's source.
Anyone redistributing an assembled environment or container must review and
satisfy the wheel and CUDA runtime licenses.

## HOTC data and modal annotations

The official HOTC dataset is not redistributed here. Obtain it from the
organizer-listed sources and follow the applicable competition and dataset
terms.

The separately hosted
<https://huggingface.co/datasets/ryo818/HOTC2026-Modal> dataset declares CC BY
4.0 for its author-created annotation layer and documentation. Its dataset card
states that this license does not cover official HOTC imagery or other
organizer-provided data. This Git repository links to that dataset but does not
redistribute it.
