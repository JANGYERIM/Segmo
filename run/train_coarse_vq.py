import os
import torch
import numpy as np
from collections import OrderedDict
from os.path import join as pjoin
from torch.utils.data import DataLoader

from models.vq.model import RVQVAE
from models.coarse_vq.coarse_vq import CoarseVQ

from options.train_option import TrainT2MOptions
from utils.get_opt import get_opt
from utils.fixseed import fixseed
from data.t2m_dataset import Text2MotionDataset, collate_fn


def load_vq_model(opt, dim_pose):
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    vq_model = RVQVAE(vq_opt, dim_pose,
                      vq_opt.nb_code, vq_opt.code_dim, vq_opt.output_emb_width,
                      vq_opt.down_t, vq_opt.stride_t, vq_opt.width, vq_opt.depth,
                      vq_opt.dilation_growth_rate, vq_opt.vq_act, vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name,
                             'model', 'net_best_fid.tar'), map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    vq_model.eval()
    for p in vq_model.parameters():
        p.requires_grad = False
    print(f'VQ Model {opt.vq_name} loaded and frozen.')
    return vq_model, vq_opt


if __name__ == '__main__':
    opt = TrainT2MOptions().parse()
    fixseed(opt.seed)

    opt.device = torch.device('cpu' if opt.gpu_id == -1 else f'cuda:{opt.gpu_id}')
    torch.autograd.set_detect_anomaly(False)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    os.makedirs(opt.model_dir, exist_ok=True)

    if opt.dataset_name == 't2m':
        opt.data_root = '/data4/local_datasets/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
        opt.seg_caption_dir = pjoin(opt.data_root, 'SegmentedCaption')
        dim_pose = 263
    else:
        raise NotImplementedError(f'Dataset {opt.dataset_name} not supported.')

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    # --- VQ-VAE (frozen) ---
    vq_model, vq_opt = load_vq_model(opt, dim_pose)
    vq_model.to(opt.device)

    # --- Dataset ---
    train_split = pjoin(opt.data_root, 'train.txt')
    val_split   = pjoin(opt.data_root, 'val.txt')

    train_dataset = Text2MotionDataset(opt, mean, std, train_split,
                                       seg_caption_dir=opt.seg_caption_dir)
    val_dataset   = Text2MotionDataset(opt, mean, std, val_split,
                                       seg_caption_dir=opt.seg_caption_dir)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=opt.batch_size, shuffle=False,
                              num_workers=4, collate_fn=collate_fn, drop_last=False)

    # --- CoarseVQ ---
    coarse_vq = CoarseVQ(
        code_dim=vq_opt.code_dim,
        nb_coarse_code=256,
        coarse_dim=256,
        clip_dim=512,
        unit_length=opt.unit_length,
        clip_version='ViT-B/32',
        device=opt.device,
    ).to(opt.device)

    optimizer = torch.optim.AdamW(
        [p for p in coarse_vq.parameters() if p.requires_grad],
        lr=2e-4, weight_decay=1e-5
    )

    best_val_loss = float('inf')
    log_every = opt.log_every

    for epoch in range(opt.max_epoch):
        coarse_vq.train()
        logs = OrderedDict(total=0., commit=0., codebook=0., align=0.)

        for it, batch in enumerate(train_loader):
            captions, motions, m_lens, seg_captions = batch
            motions = motions.to(opt.device).float()
            m_lens  = m_lens.to(opt.device)

            # fine tokens from frozen VQ-VAE
            with torch.no_grad():
                _, all_codes = vq_model.encode(motions)
                # all_codes: (b, seqlen, code_dim) 첫 번째 quantizer만 사용
                fine_tokens = all_codes[0].permute(0, 2, 1)  # (b, seqlen, code_dim)

            optimizer.zero_grad()
            loss, commit, codebook, align, _ = coarse_vq(fine_tokens, m_lens, seg_captions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(coarse_vq.parameters(), 0.5)
            optimizer.step()

            logs['total']    += loss.item()
            logs['commit']   += commit
            logs['codebook'] += codebook
            logs['align']    += align

            if (it + 1) % log_every == 0:
                avg = {k: v / log_every for k, v in logs.items()}
                print(f'ep {epoch} it {it+1:5d} | '
                      f'total={avg["total"]:.4f}  commit={avg["commit"]:.4f}  '
                      f'codebook={avg["codebook"]:.4f}  align={avg["align"]:.4f}')
                logs = OrderedDict(total=0., commit=0., codebook=0., align=0.)

        # --- validation ---
        coarse_vq.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                captions, motions, m_lens, seg_captions = batch
                motions = motions.to(opt.device).float()
                m_lens  = m_lens.to(opt.device)
                _, all_codes = vq_model.encode(motions)
                fine_tokens = all_codes[0].permute(0, 2, 1)
                loss, *_ = coarse_vq(fine_tokens, m_lens, seg_captions)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        print(f'[Epoch {epoch}] val_loss={val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'coarse_vq': coarse_vq.state_dict(), 'ep': epoch},
                       pjoin(opt.model_dir, 'coarse_vq_best.tar'))
            print(f'  → best model saved (val_loss={val_loss:.4f})')

        torch.save({'coarse_vq': coarse_vq.state_dict(), 'ep': epoch},
                   pjoin(opt.model_dir, 'coarse_vq_latest.tar'))
