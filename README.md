# SupConWI-RL: wafer inspection with reinforcement learning enhanced by supervised contrastive learning

Official implementation of *SupConWI-RL: wafer inspection with reinforcement learning enhanced by supervised contrastive learning* by Aleksandr Dekhovich and Oleg Soloviev, ICCV 2025 Workshops (VISION'25).

[Paper (CVF Open Access)](https://openaccess.thecvf.com/content/ICCV2025W/VISION'25/html/Dekhovich_SupConWI-RL_wafer_inspection_with_reinforcement_learning_enhanced_by_supervised_contrastive_ICCVW_2025_paper.html)

## Abstract

Monitoring manufacturing processes plays an important role in chip production. Current state-of-the-art approaches use the entire surface to classify defects with CNN- or Transformer-based models, resulting in considerable measurement costs. Therefore, new advanced techniques are required to reduce the cost of inspection. In this work, we advocate for the reinforcement learning-based feedback loop with a classifier trained with supervised contrastive loss. In contrast to previous works in this manner, our approach is not limited to only one type of defect but can identify multiple defects on one wafer. We tested our algorithm on the publicly available WM-811k and MixedWM38 datasets, showing a significant reduction in scanning time compared to CNN-based approaches while maintaining similar accuracy. We demonstrate the reduction of up to 40% in costs associated with wafer scanning in defect classification tasks, even if multiple defects are on the surface. Moreover, we demonstrate that in the multi-defect scenario, the trained model can be directly used to detect outliers, requiring only about 12.5% of the surface to find at least one type of defect.

## Method overview

SupConWI-RL scans a wafer map patch by patch instead of measuring the entire surface:

- **SupConRNN** (a GRU/LSTM classifier) processes one patch at a time and is trained with a combined classification + supervised contrastive loss, giving it better-separated embeddings for defect recognition.
- **confidRNN** watches SupConRNN's hidden state after every patch and predicts whether the current prediction is already trustworthy enough to stop scanning.
- A **PPO-trained controller** (the "agent") decides which patch to scan next, given the classifier's current hidden state and a mask of already-scanned patches, so as to reach a confident prediction in as few patches as possible.
- A **hybrid** fallback uses a plain CNN (ResNet-34) for the rare wafers where confidRNN never becomes confident even after a full scan.
- An **ensemble** of `M` independently trained models (classifier + confidnet + policy, each with its own `--seed`) can be combined at evaluation time to improve accuracy; see [Ensemble evaluation](#ensemble-evaluation) below.

The same trained model can also be used, without retraining, as an anomaly detector: instead of waiting to identify *every* defect on the wafer, scanning stops as soon as the presence of *any* defect (or the absence of all of them) is confident.

## Repository structure

```
src/
├── main.py                        # entry point: wires everything together based on parser.py flags
├── parser.py                      # command-line arguments
├── dataset/
│   ├── wm811k.py                  # WM-811k dataset (single-defect, multi-class)
│   └── mixedwm38.py               # MixedWM38 dataset (multi-defect, multi-label, K=8 defect types)
├── models/
│   ├── classifier/                # standalone classifiers (ResNet, RNN/SupConRNN) for direct supervised training
│   ├── confidence/                # standalone confidence estimator (RNNSelfConfid) for direct supervised training
│   └── patch_selection/           # the RL environment/agent versions used during sequential scanning:
│       ├── my_rnn.py              #   classifier wrapped as a single-step environment component
│       ├── confidnet.py           #   confidence estimator wrapped as a single-step environment component
│       ├── environment.py         #   gym.Env wrapping classifier + confidnet (single model)
│       ├── environment_ensemble.py#   same, for M models combined
│       ├── ppo.py                 #   PPO controller (single model)
│       ├── ppo_ensemble.py        #   PPO controller ensemble (inference-only, see below)
│       ├── reinforce.py           #   REINFORCE controller (single model)
│       └── cells.py               #   custom GRU/LSTM cell implementations used by the RL-side models
├── trainer/                        # training loops for each stage (CNN, RNN, confidnet, PPO, REINFORCE)
└── utils/                          # losses, data loading, evaluation/inference loops
```

## Datasets

- **WM-811k**: publicly available at [mirlab.org](http://mirlab.org/dataSet/public/); pass the path to the labeled dataset (`.pkl`) via `--data_path`.
- **MixedWM38**: publicly available `.npz` file containing wafer maps (`arr_0`) and multi-label defect annotations (`arr_1`); pass its path via `--data_path`.

By default, `--data_name` is `mixedwm38` and `--data_path` points at a placeholder location — update `--data_path` to wherever you've placed the dataset file on your machine.

## Installation

```bash
git clone https://github.com/adekhovich/SupConWI-RL.git
cd SupConWI-RL
pip install -r requirements.txt
```

## Usage

All commands are run from `src/`.

**Train the classifier** (with supervised contrastive loss) on MixedWM38:

```bash
python3 main.py --data_name mixedwm38 --data_path /path/to/MixedWM38.npz \
    --train_classifier --classifier_name lstm --supcon
```

**Train the confidence estimator**, using an already-trained classifier:

```bash
python3 main.py --data_name mixedwm38 --data_path /path/to/MixedWM38.npz \
    --classifier_name lstm --supcon --train_confidnet
```

**Train the PPO patch selector**, using an already-trained classifier + confidnet:

```bash
python3 main.py --data_name mixedwm38 --data_path /path/to/MixedWM38.npz \
    --classifier_name lstm --supcon --train_patch_selector
```

**Evaluate with the hybrid approach** (fall back to a CNN, e.g. ResNet-34, for wafers that need a full scan):

```bash
python3 main.py --data_name mixedwm38 --data_path /path/to/MixedWM38.npz \
    --classifier_name lstm --supcon --hybrid --aux_model_name resnet34
```

### Ensemble evaluation

The ensemble is **inference-only**: it combines `num_models` independently trained single-model pipelines rather than being trained jointly. To reproduce it:

1. Train `num_models` single models (classifier, confidnet, and policy) as above, once per seed:
   ```bash
   for seed in 0 1 2 3 4; do
       python3 main.py --seed $seed --train_classifier --train_confidnet --train_patch_selector \
           --classifier_name lstm --supcon --data_path /path/to/MixedWM38.npz
   done
   ```
2. Evaluate the ensemble by loading all `num_models` checkpoints and combining their decisions:
   ```bash
   python3 main.py --ensemble --num_models 5 --classifier_name lstm --supcon \
       --data_path /path/to/MixedWM38.npz
   ```

For WM-811k, use `--data_name wm811k --data_path /path/to/WM811K_labeled.pkl`.

## Results (from the paper)

On WM-811k, compared to SWI ([Dekhovich et al., 2025](https://doi.org/10.1016/j.eswa.2025.126996)), Ensemble SupConWI-RL reaches 85.69% accuracy using 38.0/64 patches on average (vs. 40.7/64 for SWI), while a full ResNet-34 scan needs all 64 patches for 85.52%.

On MixedWM38 (the multi-defect setting), Ensemble SupConWI-RL reaches 98.45% accuracy using 102.1/169 patches on average (a ≈40% reduction in scanning cost), versus 98.79% for a full ResNet-34 scan using all 169 patches.

In the anomaly-detection setting (flagging a wafer as defective as soon as *any* one defect is confidently identified), the model reaches 100% accuracy using on average only 21.1/169 patches (12.5% of the surface).

See the paper for the full ablation study (contribution of the contrastive loss, the ensemble, and the hybrid CNN fallback) and additional baselines.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{dekhovich2025supconwi,
  title={SupConWI-RL: Wafer inspection with reinforcement learning enhanced by supervised contrastive learning},
  author={Dekhovich, Aleksandr and Soloviev, Oleg},
  booktitle={2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  pages={1396--1405},
  year={2025},
  organization={IEEE}
}
```

## Acknowledgement

This project is supported by the Chips Joint Undertaking and its members, including the top-up funding by RVO (The Netherlands Enterprise Agency).
