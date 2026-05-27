"""
Generate motions for 300 random test set samples.

Usage:
    python run/gen_t2m.py --name MTRANS_V1 --res_name rtrans_V1 --gpu_id 0 --ext sample_300
"""

import os
import shutil
import json
import random
import subprocess
import tempfile
from os.path import join as pjoin

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch.distributions.categorical import Categorical

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE, LengthEstimator
from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from utils.motion_process import recover_from_ric
from utils.paramUtil import t2m_kinematic_chain

clip_version = 'ViT-B/32'
DATASET_DIR = '/data4/local_datasets/HumanML3D'
NUM_SAMPLES = 300


# ─── Model loaders ────────────────────────────────────────────────────────────

def load_vq_model(vq_opt):
    vq_model = RVQVAE(vq_opt,
                      vq_opt.dim_pose, vq_opt.nb_code, vq_opt.code_dim,
                      vq_opt.output_emb_width, vq_opt.down_t, vq_opt.stride_t,
                      vq_opt.width, vq_opt.depth, vq_opt.dilation_growth_rate,
                      vq_opt.vq_act, vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name,
                             vq_opt.name, 'model', 'net_best_fid.tar'), map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt


def load_trans_model(model_opt, opt, which_model):
    t2m_transformer = MaskTransformer(
        code_dim=model_opt.code_dim, cond_mode='text',
        latent_dim=model_opt.latent_dim, ff_size=model_opt.ff_size,
        num_layers=model_opt.n_layers, num_heads=model_opt.n_heads,
        dropout=model_opt.dropout, clip_dim=512,
        cond_drop_prob=model_opt.cond_drop_prob,
        clip_version=clip_version, opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name,
                             model_opt.name, 'model', which_model), map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all(k.startswith('clip_model.') for k in missing_keys)
    print(f'Loading Transformer {model_opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer


def load_res_model(res_opt, vq_opt, opt):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(
        code_dim=vq_opt.code_dim, cond_mode='text',
        latent_dim=res_opt.latent_dim, ff_size=res_opt.ff_size,
        num_layers=res_opt.n_layers, num_heads=res_opt.n_heads,
        dropout=res_opt.dropout, clip_dim=512,
        shared_codebook=vq_opt.shared_codebook,
        cond_drop_prob=res_opt.cond_drop_prob,
        share_weight=res_opt.share_weight,
        clip_version=clip_version, opt=res_opt)
    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name,
                             res_opt.name, 'model', 'net_best_fid.tar'), map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all(k.startswith('clip_model.') for k in missing_keys)
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer


def load_len_estimator(opt):
    model = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name,
                             'length_estimator', 'model', 'finest.tar'), map_location=opt.device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from epoch {ckpt["epoch"]}!')
    return model


# ─── Data helpers ─────────────────────────────────────────────────────────────

def read_full_captions(motion_id):
    """Return list of full caption strings from texts/{motion_id}.txt."""
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


def read_segmented_captions(motion_id, cap_idx_1based):
    """Return list of segment strings for the N-th caption (1-indexed)."""
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


def get_token_length(motion_id):
    """Return token length (frame_len // 4) from npy, or None if not found."""
    for name in [motion_id, 'M' + motion_id]:
        path = pjoin(DATASET_DIR, 'new_joint_vecs', name + '.npy')
        if os.path.exists(path):
            return np.load(path).shape[0] // 4
    return None


# ─── Translation ──────────────────────────────────────────────────────────────

def translate_batch(texts):
    """Translate a list of English texts to Korean via googletrans."""
    try:
        from googletrans import Translator
        translator = Translator()
        results = []
        for text in texts:
            try:
                results.append(translator.translate(text, src='en', dest='ko').text)
            except Exception:
                results.append(None)
        return results
    except ImportError:
        return [None] * len(texts)


# ─── Video helpers ────────────────────────────────────────────────────────────

def render_motion(joints, save_path, kinematic_chain, fps=20, radius=4):
    """Render motion to mp4 with no title text, motion filling the frame."""
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=(6, 6))
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


def hstack_videos(video_paths, output_path):
    """Stack videos horizontally side by side using ffmpeg hstack."""
    n = len(video_paths)
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
            os.remove(p)


# ─── Generation helpers ───────────────────────────────────────────────────────

def gen_motions(captions, token_lens, t2m_transformer, res_model, vq_model,
                opt, inv_transform):
    """Generate and decode motions. Returns (joints_list, token_lens_cpu)."""
    with torch.no_grad():
        mids = t2m_transformer.generate(
            captions, token_lens,
            timesteps=opt.time_steps, cond_scale=opt.cond_scale,
            temperature=opt.temperature, topk_filter_thres=opt.topkr,
            gsample=opt.gumbel_sample)
        mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
        pred = vq_model.forward_decoder(mids).detach().cpu().numpy()

    pred = inv_transform(pred)
    tlens = token_lens.cpu().numpy()
    joints_list = []
    for i, tlen in enumerate(tlens):
        joint_data = pred[i, :tlen * 4]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
        joints_list.append(joint)
    return joints_list


def estimate_token_lens(captions, t2m_transformer, length_estimator, device):
    with torch.no_grad():
        text_emb = t2m_transformer.encode_text(captions)
        pred_dis = length_estimator(text_emb)
        token_lens = Categorical(F.softmax(pred_dis, dim=-1)).sample()
    return token_lens.to(device)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else f"cuda:{opt.gpu_id}")
    dim_pose = 263  # HumanML3D

    result_dir = pjoin('./generation', opt.ext)
    os.makedirs(result_dir, exist_ok=True)

    # ── Load models ──
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_opt = get_opt(pjoin(root_dir, 'opt.txt'), device=opt.device)

    vq_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name,
                            model_opt.vq_name, 'opt.txt'), device=opt.device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)
    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name,
                             opt.res_name, 'opt.txt'), device=opt.device)
    res_model = load_res_model(res_opt, vq_opt, opt)
    assert res_opt.vq_name == model_opt.vq_name

    t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')
    length_estimator = load_len_estimator(model_opt)

    for m in [t2m_transformer, vq_model, res_model, length_estimator]:
        m.eval()
        m.to(opt.device)

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name,
                          model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name,
                         model_opt.vq_name, 'meta', 'std.npy'))
    inv_transform = lambda data: data * std + mean

    # ── Sample 300 test IDs ──
    with open(pjoin(DATASET_DIR, 'test.txt')) as f:
        all_test_ids = [l.strip() for l in f if l.strip()]

    valid_ids = [mid for mid in all_test_ids if get_token_length(mid) is not None]
    random.seed(opt.seed)
    sampled_ids = random.sample(valid_ids, min(NUM_SAMPLES, len(valid_ids)))
    print(f"Sampled {len(sampled_ids)} / {len(valid_ids)} valid test IDs")

    kinematic_chain = t2m_kinematic_chain
    all_json = {}
    json_path = pjoin(result_dir, 'captions.json')

    for idx, motion_id in enumerate(sampled_ids):
        print(f"\n[{idx + 1}/{len(sampled_ids)}] {motion_id}")
        token_len = get_token_length(motion_id)
        full_captions = read_full_captions(motion_id)
        if not full_captions:
            print("  No captions, skipping")
            continue

        json_entry = {"captions": []}

        for cap_idx, full_cap in enumerate(full_captions, 1):
            seg_captions = read_segmented_captions(motion_id, cap_idx)
            if not seg_captions:
                print(f"  caption {cap_idx}: no segments, skipping")
                continue

            temp_videos = []

            # ── Full caption → real token length ──
            tlen = torch.LongTensor([token_len]).to(opt.device)
            [full_joint] = gen_motions([full_cap], tlen,
                                        t2m_transformer, res_model, vq_model,
                                        opt, inv_transform)
            tmp = tempfile.mktemp(suffix='.mp4')
            render_motion(full_joint, tmp, kinematic_chain)
            temp_videos.append(tmp)

            # ── Segment captions → estimated lengths ──
            seg_tlens = estimate_token_lens(seg_captions, t2m_transformer,
                                            length_estimator, opt.device)
            seg_joints = gen_motions(seg_captions, seg_tlens,
                                      t2m_transformer, res_model, vq_model,
                                      opt, inv_transform)
            for seg_joint in seg_joints:
                tmp = tempfile.mktemp(suffix='.mp4')
                render_motion(seg_joint, tmp, kinematic_chain)
                temp_videos.append(tmp)

            # ── Hstack & save video ──
            out_path = pjoin(result_dir, f'{motion_id}_{cap_idx}.mp4')
            hstack_videos(temp_videos, out_path)
            cleanup(temp_videos)
            print(f"  caption {cap_idx}: saved → {out_path}")

            # ── Translate ──
            all_texts = [full_cap] + seg_captions
            translations = translate_batch(all_texts)

            json_entry["captions"].append({
                "full": full_cap,
                "full_ko": translations[0],
                "segments": [
                    {"text": seg, "text_ko": tr}
                    for seg, tr in zip(seg_captions, translations[1:])
                ]
            })

        all_json[motion_id] = json_entry

        # Save JSON after each ID (incremental, crash-safe)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_json, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(all_json)} IDs saved to {result_dir}")
    print(f"Caption JSON: {json_path}")
