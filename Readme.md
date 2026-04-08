## 📌 Overview
We present a **multi-trajectory diffusion-based sampler** for solving ill-posed image restoration problems. Our method leverages the diversity of diffusion trajectories initialized from independent noise samples and introduces a **patchwise divide-and-aggregate strategy** to enhance structural exploration and reconstruction fidelity.

The core idea is to:
- Sample multiple reconstructions from a diffusion prior
- Introduce **structured diversity** via patchwise permutations/rotations
- Aggregate local reconstructions into a **globally consistent solution**

## 🧠 Motivation

Inverse problems y = Ax + Gaussian(0, sigma_y) (e.g., inpainting, super-resolution, deblurring) are inherently ill-posed:





- Multiple valid solutions \(x\) exist for a given observation \(y\)
- Diffusion models implicitly define a **posterior distribution**
- Standard samplers explore only **limited modes**

👉 Our method explicitly **explores and aggregates multiple possible reconstructions**

## 📂 Repository Structure

```bash
.
├── configs/              # Experiment configurations
├── models/               # Diffusion model definitions
├── samplers/             # Sampling algorithms
├── operators/            # Forward models (inpainting, SR, etc.)
├── utils/                # Utility functions
├── scripts/              # Training / evaluation scripts
├── outputs/              # Generated results (ignored by git)
└── README.md
