# -*- coding: utf-8 -*-
"""The list of experiments.

Every run = the paper recipe (v6, called CONTROL here) + exactly ONE change, so each change can be judged
on its own. All runs: MobileNet v1 (alpha 1.0), 192x192, image + irradiance, 10 epochs, seed 42,
ONE network and ONE forward pass per image at prediction time.
Selection uses the VALIDATION days (June 22 and 27) only; the test days are never used to choose anything.
"""
import copy
import os

# ----------------------------------------------------------------- the paper recipe (= control)
BASE = {
    "split": "splits_v5",            # 10 training days, validation 22+27, test 15+24+30
    "img_size": 192,
    "alpha": 1.0,                    # MobileNet v1 width
    "dense_units": [128, 32],
    "dropout": 0.3,
    "seed": 42,
    "epochs": 10,
    "batch_size": 32,
    "lr": 5e-5,                      # Adam, cosine decay per epoch from lr (epoch 1) to min_lr (epoch 10)
    "min_lr": 5e-6,
    "lds": {"bins": 50, "sigma": 2.0, "power": 0.75, "clip": 4.0},    # sample weights (Yang et al., ICML 2021)
    "aug": {"hflip": True, "vflip": True, "brightness": 0.1},
    "swa_last": 3,                   # weight averaging of the last 3 epochs (8, 9, 10)
    # ---- options that the experiments switch on (all off in the control) ----
    "loss": "mse",                   # mse | huber | bmc | dldl
    "huber_delta": 0.1,
    "bmc": {"init_sigma": 0.1, "sigma_lr": 1e-2},
    "dldl": {"bins": 101, "sigma": 0.02, "l1_weight": 1.0},
    "ranksim": None,                 # {"gamma": .., "lam": ..}
    "cmixup": None,                  # {"alpha": .., "bandwidth": ..}
    "warmup_epochs": 0,
    "llrd": None,                    # {"factor": 2.6}
    "distill": None,                 # {"alpha": 0.5, "teacher": "<run name>"}
    "attention": None,               # None | "eca" (ECA channel attention after the last MobileNet block)
    "panel_mask": False,             # True = grey out everything except the soiled panel (train, test, prediction)
    "coverage_input": False,         # True = soiled-area ratio, measured from the image, is an extra feature
    "coverage_head": False,          # True = second output: share of the panel covered by soiling (multi-task)
    "coverage_weight": 1.0,          # weight of the soiled-area loss
    "panel_features": None,          # e.g. ["panel_index", "sun"]: image-processing features next to the irradiance
    "feature_dropout": 0.0,          # share of training samples in which the panel index is hidden (-1)
    "aux_targets": None,             # extra training-only targets (soiling heatmap + regional index), see train_one.py
}

# ----------------------------------------------------------------- one change per run
SCREEN = [
    ("00_control", "Paper recipe (v6): weighted MSE + LDS, 10 epochs, cosine LR", {}),
    ("01_balanced_mse",
     "Balanced MSE, batch Monte-Carlo form (Ren et al., CVPR 2022). Replaces LDS weighting + MSE",
     {"loss": "bmc", "lds": None}),
    ("02_ranksim",
     "RankSim feature-ranking regulariser (Gong et al., ICML 2022) on the 32-unit features",
     {"ranksim": {"gamma": 0.1, "lam": 2.0}}),
    ("03_cmixup",
     "C-Mixup: mix images whose labels are close (Yao et al., NeurIPS 2022)",
     {"cmixup": {"alpha": 2.0, "bandwidth": 0.05}}),
    ("04_dldl",
     "Label-distribution learning DLDL-v2 (Gao et al., TIP 2017; IJCAI 2018): 101-bin softmax, "
     "KL + L1 on the expected value",
     {"loss": "dldl"}),
    ("05_huber", "Huber loss, delta 0.10 (Huber, 1964)", {"loss": "huber"}),
    ("06_warmup_llrd",
     "1-epoch LR warm-up (Goyal et al., 2017) + layer-wise LR, /2.6 per group (Howard & Ruder, ACL 2018)",
     {"warmup_epochs": 1, "llrd": {"factor": 2.6}}),
    ("07_self_distill",
     "Born-again self-distillation (Furlanello et al., ICML 2018): teacher = 00_control (same network)",
     {"distill": {"alpha": 0.5, "teacher": "00_control"}}),
]

# changes that replace the loss cannot be combined with each other; the best one is kept
LOSS_GROUP = {"01_balanced_mse", "04_dldl", "05_huber"}

COMBO = "08_combo"
FINAL_TEACHER = "09_final_teacher_12days"
FINAL = "10_final_12days"


def config_for(changes):
    cfg = copy.deepcopy(BASE)
    for k, v in changes.items():
        cfg[k] = copy.deepcopy(v)
    return cfg


def screen_changes():
    return {name: ch for name, _, ch in SCREEN}


def descriptions():
    return {name: d for name, d, _ in SCREEN}


def merge_changes(names):
    """Changes of several winning runs in one config. Only one loss-type change is allowed."""
    ch = screen_changes()
    merged = {}
    for n in names:
        merged.update(copy.deepcopy(ch[n]))
    return merged


def smoke_overrides(cfg):
    """Tiny settings used by --smoke (checks that the code runs; numbers mean nothing)."""
    cfg = copy.deepcopy(cfg)
    cfg.update({"epochs": 2, "batch_size": 8, "swa_last": 2, "max_images": int(os.environ.get("SMOKE_MAX", 48))})
    return cfg
