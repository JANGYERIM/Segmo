"""
Visualize GT motion alongside its GT segment crops for training samples.

For each sampled motion ID (train split, 3+ segments):
  - GT        : full GT motion × 3
  - GT_seg1~3 : equal-time crops from GT, each tiled to match GT×3 length

Side-by-side hstack video saved to generation/<ext>/.
JSON saved with motion_id, text, seg_cap1/2/3.

Usage:
    python run/viz_gt_segments.py --ext gt_seg_viz --gpu_id 0
"""

import os
import json
import random
import tempfile
import subprocess
from os.path import join as pjoin

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch

from options.eval_option import EvalT2MOptions
from utils.fixseed import fixseed
from utils.motion_process import recover_from_ric
from utils.paramUtil import t2m_kinematic_chain

DATASET_DIR = '/data4/local_datasets/HumanML3D'
NUM_SAMPLES = 50
MIN_SEGS = 3


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_train_ids():
    with open(pjoin(DATASET_DIR, 'train.txt')) as f:
        return [l.strip() for l in f if l.strip()]


def read_seg_captions(motion_id, cap_idx_1based):
    """Return list of segment text strings (split on '#') or []."""
    path = pjoin(DATASET_DIR, 'SegmentedCaption', f'{motion_id}_{cap_idx_1based}.txt')
    if not os.path.exists(path):
        return []
    segs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                segs.append(line.split('#')[0])
    return segs


def read_full_captions(motion_id):
    """Return list of full caption strings."""
    path = pjoin(DATASET_DIR, 'texts', motion_id + '.txt')
    if not os.path.exists(path):
        return []
    captions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                captions.append(line.split('#')[0])
    return captions


def load_gt_motion(motion_id):
    """Return raw motion array or None."""
    for name in [motion_id, 'M' + motion_id]:
        path = pjoin(DATASET_DIR, 'new_joint_vecs', name + '.npy')
        if os.path.exists(path):
            return np.load(path).astype(np.float32)
    return None


def motion_to_joints(data, n_frames=None):
    """Recover 3D joints from HumanML3D feature vector."""
    if n_frames is not None:
        data = data[:n_frames]
    return recover_from_ric(torch.from_numpy(data).float(), 22).numpy()


def seg_boundaries(token_len, n_seg):
    """Return list of (si_frame, ei_frame) in original frame space (×4)."""
    bounds = []
    for idx in range(n_seg):
        si_tok = idx * token_len // n_seg
        ei_tok = (idx + 1) * token_len // n_seg
        ei_tok = max(ei_tok, si_tok + 1)
        bounds.append((si_tok * 4, ei_tok * 4))
    return bounds


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_motion(joints, save_path, kinematic_chain, fps=20, radius=4):
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=(4, 4))
    ax = p3.Axes3D(fig)
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([0, radius])
    ax.set_zlim3d([0, radius])
    ax.grid(b=False)

    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [[minx, miny, minz], [minx, miny, maxz],
                 [maxx, miny, maxz], [maxx, miny, minz]]
        xz = Poly3DCollection([verts])
        xz.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz)

    def update(idx):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[idx, 0], MAXS[0] - trajec[idx, 0], 0,
                     MINS[2] - trajec[idx, 1], MAXS[2] - trajec[idx, 1])
        if idx > 1:
            ax.plot3D(trajec[:idx, 0] - trajec[idx, 0],
                      np.zeros_like(trajec[:idx, 0]),
                      trajec[:idx, 1] - trajec[idx, 1],
                      linewidth=1.0, color='blue')
        for i, (chain, color) in enumerate(zip(kinematic_chain, colors)):
            lw = 4.0 if i < 5 else 2.0
            ax.plot3D(data[idx, chain, 0], data[idx, chain, 1], data[idx, chain, 2],
                      linewidth=lw, color=color)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=len(data), interval=1000 / fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()


def tile_to_length(joints, target_frames):
    """Repeat joints array until it reaches target_frames, then crop."""
    if len(joints) >= target_frames:
        return joints[:target_frames]
    reps = (target_frames // len(joints)) + 1
    tiled = np.concatenate([joints] * reps, axis=0)
    return tiled[:target_frames]


def hstack_videos(video_paths, output_path):
    n = len(video_paths)
    if n == 1:
        import shutil
        shutil.copy(video_paths[0], output_path)
        return
    inputs = sum([['-i', p] for p in video_paths], [])
    filter_complex = f'hstack=inputs={n}[out]'
    subprocess.run(
        ['ffmpeg', '-y'] + inputs +
        ['-filter_complex', filter_complex, '-map', '[out]', output_path],
        check=True, capture_output=True
    )


def cleanup(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    result_dir = pjoin('./generation', opt.ext)
    os.makedirs(result_dir, exist_ok=True)

    kinematic_chain = t2m_kinematic_chain

    # ── Build candidate list: train IDs with 3+ segments ──
    train_ids = set(load_train_ids())
    seg_dir = pjoin(DATASET_DIR, 'SegmentedCaption')

    candidates = {}  # motion_id -> list of (cap_idx_str, [seg_texts])
    for fname in sorted(os.listdir(seg_dir)):
        if not fname.endswith('.txt'):
            continue
        base = fname[:-4]
        parts = base.rsplit('_', 1)
        if len(parts) != 2:
            continue
        mid, cap_idx = parts[0], parts[1]
        if mid not in train_ids:
            continue
        with open(pjoin(seg_dir, fname)) as f:
            segs = [l.strip().split('#')[0] for l in f if l.strip()]
        if len(segs) >= MIN_SEGS:
            candidates.setdefault(mid, []).append((cap_idx, segs))

    print(f"Train IDs with {MIN_SEGS}+ segments: {len(candidates)}")

    # Filter: GT motion must exist
    valid = {mid: caps for mid, caps in candidates.items()
             if load_gt_motion(mid) is not None}
    print(f"Valid (GT exists): {len(valid)}")

    random.seed(opt.seed)
    sampled = random.sample(list(valid.keys()), min(NUM_SAMPLES, len(valid)))
    print(f"Sampled: {len(sampled)}")

    all_json = {}
    json_path = pjoin(result_dir, 'gt_segments.json')

    for i, motion_id in enumerate(sampled):
        print(f"\n[{i+1}/{len(sampled)}] {motion_id}")

        raw = load_gt_motion(motion_id)
        if raw is None:
            continue
        token_len = raw.shape[0] // 4

        full_captions = read_full_captions(motion_id)

        # Use first cap_idx that has 3+ segments
        cap_idx_str, seg_texts = valid[motion_id][0]
        cap_idx = int(cap_idx_str)
        full_text = full_captions[cap_idx - 1] if cap_idx <= len(full_captions) else ""

        n_seg = len(seg_texts)
        bounds = seg_boundaries(token_len, n_seg)

        # GT full joints × 3
        gt_frames = min(token_len * 4, raw.shape[0])
        gt_joints = motion_to_joints(raw, n_frames=gt_frames)
        target_frames = len(gt_joints) * 3
        gt_rep = np.concatenate([gt_joints] * 3, axis=0)

        temp_videos = []

        # Render GT
        tmp = tempfile.mktemp(suffix='.mp4')
        render_motion(gt_rep, tmp, kinematic_chain)
        temp_videos.append(tmp)

        # Render each segment (tiled to target_frames)
        seg_joints_list = []
        for si_f, ei_f in bounds:
            si_f = min(si_f, raw.shape[0] - 1)
            ei_f = min(ei_f, raw.shape[0])
            seg_raw = raw[si_f:ei_f]
            seg_j = motion_to_joints(seg_raw)
            seg_j = tile_to_length(seg_j, target_frames)
            seg_joints_list.append(seg_j)

            tmp = tempfile.mktemp(suffix='.mp4')
            render_motion(seg_j, tmp, kinematic_chain)
            temp_videos.append(tmp)

        out_path = pjoin(result_dir, f'{motion_id}_{cap_idx_str}.mp4')
        try:
            hstack_videos(temp_videos, out_path)
            print(f"  saved → {out_path}")
        except Exception as e:
            print(f"  hstack failed: {e}")
        finally:
            cleanup(temp_videos)

        entry = {
            "motion_id": motion_id,
            "text": full_text,
        }
        for j, seg_t in enumerate(seg_texts, 1):
            entry[f"seg_cap{j}"] = seg_t

        all_json[motion_id] = entry

        # Incremental save
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_json, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(all_json)} entries → {json_path}")
