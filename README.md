# PLA-AASP Net

PyTorch implementation of **PLA-AASP Net: Multi-Frame Super-Resolution Method Based on Phase-Lattice Aggregation and Alignment-Aware State Propagation**.

## Abstract

Multi-frame super-resolution (MFSR) recovers high-resolution (HR) images by fusing complementary information from multiple observations. However, feature fusion after inter-frame alignment can obscure sub-pixel sampling differences, while residual alignment errors may propagate beyond misaligned regions through spatial state updates. To address these issues, we propose an MFSR network based on phase-lattice aggregation and alignment-aware state propagation, namely PLA-AASP Net. First, a shared encoder and optical-flow-guided alignment produce aligned features, alignment confidence, and effective displacements, providing the content, reliability, and sampling-position information needed for subsequent aggregation. Then, the phase-lattice aggregation module uses fractional effective displacements to assign cross-frame residuals to fixed phase slots and aggregates them according to content relevance and alignment confidence, preserving slot indices through phase interaction for subsequent HR sub-pixel reconstruction. Next, the alignment-aware state propagation module combines fusion reliability, total effective evidence, and local edge information to regulate current evidence writing and historical state retention separately, reducing the propagation of unreliable information beyond misaligned regions. Finally, the reconstruction module combines spatial features with the retained phase-slot features through shared projection and sub-pixel rearrangement to recover the HR image. Experimental results on SBSR and RealBSR at ×4 and ×8 upscaling show that PLA-AASP Net achieves the best results in 10 of 12 metric comparisons and the second-best results in the remaining two among the compared methods. 

## Project Structure

```text
PLA-AASP-Net/
|-- models/
|   |-- common.py       # Shared layers and WKV scan
|   |-- alignment.py    # Feature encoding and alignment
|   |-- pla.py          # Phase-lattice aggregation
|   |-- aasp.py         # Alignment-aware state propagation
|   |-- psi.py          # Phase-spatial interaction
|   `-- main_network.py # Complete network
|-- config.py           # Paths and parameters
|-- dataset.py          # Data loading
|-- loss.py             # Training loss
|-- metrics.py          # PSNR, SSIM, and LPIPS
|-- utils.py            # Helper functions
|-- train.py            # Training
|-- test.py             # Evaluation
|-- main.py             # Entry point
`-- requirements.txt    # Dependencies
```

## Installation

Python 3.10 and PyTorch 2.0.0.

```bash
conda create -n pla_aasp python=3.10
conda activate pla_aasp
pip install -r requirements.txt
```

## Dataset

Place paired data in the following folders, or change the paths in `config.py`:

```text
dataset/
|-- folder1/   # Training LR bursts
|-- folder3/   # Training HR images
|-- folder5/   # Testing LR bursts
`-- folder6/   # Testing HR images
```

We provide the dataset of our work. You can download it by the following link: [dataset](https://pan.baidu.com/s/1jQBMkcJPPafd8hDNj323CQ?pwd=3114 ) code:3114.

## Usage

Set parameters in `config.py`. For fresh training, place the [pretrained SPyNet weights](https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth) in `checkpoints/`.

```bash
python main.py --mode train
python main.py --mode test --checkpoint checkpoints/best.pth
```

Checkpoints are saved in `checkpoints/` and test results in `results/`.
