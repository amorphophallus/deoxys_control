#!/usr/bin/env python3
"""Plot index_tip - thumb_tip and keypoint components over time from npz."""
import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "npz__trans_output",
        ),
    )
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    return parser.parse_args()


def _find_latest_npz(base_dir: str) -> str:
    try:
        entries = [
            entry
            for entry in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, entry))
        ]
    except FileNotFoundError:
        return os.path.join(base_dir, "hand_joints_21_base_mano.npz")

    candidates = []
    for entry in entries:
        match = re.match(r"\d{8}_\d{6}$", entry)
        if not match:
            continue
        path = os.path.join(base_dir, entry, "hand_joints_21_base_mano.npz")
        if os.path.isfile(path):
            candidates.append((entry, path))
    if not candidates:
        return os.path.join(base_dir, "hand_joints_21_base_mano.npz")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def main():
    args = parse_args()
    npz_path = args.npz
    if os.path.isdir(npz_path):
        npz_path = _find_latest_npz(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    out_dir = os.path.dirname(npz_path)

    joints = data[f"{args.hand}_joints"]
    thumb = joints[:, 4, :]
    thumb_cmc = joints[:, 1, :]
    thumb_mcp = joints[:, 2, :]
    index = joints[:, 8, :]
    wrist = joints[:, 0, :]
    valid = (
        np.isfinite(thumb).all(axis=1)
        & np.isfinite(thumb_cmc).all(axis=1)
        & np.isfinite(thumb_mcp).all(axis=1)
        & np.isfinite(index).all(axis=1)
        & np.isfinite(wrist).all(axis=1)
    )
    dirs = index[valid] - thumb[valid]
    if dirs.size == 0:
        print("No valid frames found.")
        return
    idx = np.arange(dirs.shape[0])

    plt.figure(figsize=(9, 4))
    plt.plot(idx, dirs[:, 0], label="x")
    plt.plot(idx, dirs[:, 1], label="y")
    plt.plot(idx, dirs[:, 2], label="z")
    plt.xlabel("valid frame index")
    plt.ylabel("index_tip - thumb_tip (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hand_dir_components.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(idx, index[valid][:, 0], label="x")
    plt.plot(idx, index[valid][:, 1], label="y")
    plt.plot(idx, index[valid][:, 2], label="z")
    plt.xlabel("valid frame index")
    plt.ylabel("index_tip (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "index_tip_components.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(idx, thumb[valid][:, 0], label="x")
    plt.plot(idx, thumb[valid][:, 1], label="y")
    plt.plot(idx, thumb[valid][:, 2], label="z")
    plt.xlabel("valid frame index")
    plt.ylabel("thumb_tip (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "thumb_tip_components.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(idx, thumb_cmc[valid][:, 0], label="x")
    plt.plot(idx, thumb_cmc[valid][:, 1], label="y")
    plt.plot(idx, thumb_cmc[valid][:, 2], label="z")
    plt.xlabel("valid frame index")
    plt.ylabel("thumb_cmc (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "thumb_cmc_components.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(idx, thumb_mcp[valid][:, 0], label="x")
    plt.plot(idx, thumb_mcp[valid][:, 1], label="y")
    plt.plot(idx, thumb_mcp[valid][:, 2], label="z")
    plt.xlabel("valid frame index")
    plt.ylabel("thumb_mcp (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "thumb_mcp_components.png"), dpi=150)
    plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(idx, wrist[valid][:, 0], label="x")
    plt.plot(idx, wrist[valid][:, 1], label="y")
    plt.plot(idx, wrist[valid][:, 2], label="z")
    plt.xlabel("valid frame index")
    plt.ylabel("wrist (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "wrist_components.png"), dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
