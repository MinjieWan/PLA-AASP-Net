# PLA-AASP Net

PyTorch implementation of **PLA-AASP Net: Multi-Frame Super-Resolution Method Based on Phase-Lattice Aggregation and Alignment-Aware State Propagation**.

## Abstract

PLA-AASP Net reconstructs high-resolution images from multiple low-resolution observations. A shared encoder and flow-guided alignment extract aligned features, confidence, and effective displacements. Phase-lattice aggregation (PLA) organizes complementary information by subpixel sampling positions. Alignment-aware state propagation (AASP) uses reliability and structural information to control spatial state updates. The reconstruction module combines spatial features and retained phase-slot features to recover the high-resolution image.

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
