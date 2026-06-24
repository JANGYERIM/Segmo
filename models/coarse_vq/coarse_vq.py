import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CoarseVQ(nn.Module):
    def __init__(self, code_dim=512, nb_coarse_code=256, coarse_dim=256,
                 clip_dim=512, unit_length=4, clip_version='ViT-B/32', device='cuda'):
        super().__init__()
        self.code_dim = code_dim
        self.nb_coarse_code = nb_coarse_code
        self.coarse_dim = coarse_dim
        self.unit_length = unit_length

        # fine token → coarse 공간 projection
        self.motion_proj = nn.Linear(code_dim, coarse_dim)

        # coarse codebook
        self.codebook = nn.Embedding(nb_coarse_code, coarse_dim)
        nn.init.uniform_(self.codebook.weight, -1 / nb_coarse_code, 1 / nb_coarse_code)

        # CLIP text → coarse 공간 projection
        self.text_proj = nn.Linear(clip_dim, coarse_dim)

        # contrastive temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * 0.07)

        # CLIP (frozen)
        clip_model, _ = clip.load(clip_version, device='cpu', jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip_model = clip_model

    def encode_text(self, texts, device):
        tokens = clip.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            emb = self.clip_model.encode_text(tokens).float()
        return emb  # (n, 512)

    def pool_by_segment(self, fine_tokens, m_lens, seg_captions):
        """
        fine_tokens : (b, seqlen, code_dim)
        m_lens      : (b,) 프레임 단위 실제 길이
        seg_captions: list[list[str] | None]

        Returns
        -------
        coarse_vecs : (total_segs, code_dim)  배치 내 모든 세그먼트 벡터
        seg_texts   : list[str]               대응하는 세그먼트 텍스트
        seg_counts  : list[int]               시퀀스별 세그먼트 수
        """
        coarse_list = []
        seg_texts = []
        seg_counts = []

        for b in range(fine_tokens.shape[0]):
            segs = seg_captions[b]
            n_tokens = m_lens[b].item() // self.unit_length

            if segs is None or len(segs) <= 1:
                # 세그먼트 없으면 전체를 1개 coarse token으로
                coarse_list.append(fine_tokens[b, :n_tokens].mean(dim=0, keepdim=True))
                seg_texts.append(["motion"])  # placeholder
                seg_counts.append(1)
            else:
                n_segs = len(segs)
                tokens_per_seg = n_tokens / n_segs
                seg_vecs = []
                for s in range(n_segs):
                    start = int(s * tokens_per_seg)
                    end = min(int((s + 1) * tokens_per_seg), n_tokens)
                    if start >= end:
                        start = max(0, end - 1)
                    seg_vecs.append(fine_tokens[b, start:end].mean(dim=0))
                coarse_list.append(torch.stack(seg_vecs, dim=0))  # (n_segs, code_dim)
                seg_texts.append(segs)
                seg_counts.append(n_segs)

        return torch.cat(coarse_list, dim=0), seg_texts, seg_counts

    def quantize(self, z):
        """
        z : (N, coarse_dim)
        Returns indices (N,), quantized (N, coarse_dim)
        """
        d = (torch.sum(z ** 2, dim=1, keepdim=True)
             + torch.sum(self.codebook.weight ** 2, dim=1)
             - 2 * z @ self.codebook.weight.T)  # (N, nb_coarse_code)
        indices = torch.argmin(d, dim=1)
        quantized = self.codebook(indices)
        return indices, quantized

    def forward(self, fine_tokens, m_lens, seg_captions):
        """
        fine_tokens : (b, seqlen, code_dim)  frozen VQ-VAE 출력
        m_lens      : (b,)
        seg_captions: list[list[str] | None]
        """
        device = fine_tokens.device

        # 1. 세그먼트별 pooling
        all_coarse, seg_texts_nested, seg_counts = self.pool_by_segment(
            fine_tokens, m_lens, seg_captions
        )  # (total_segs, code_dim)

        # 2. fine code_dim → coarse_dim
        z = self.motion_proj(all_coarse)  # (total_segs, coarse_dim)

        # 3. VQ quantize
        indices, quantized = self.quantize(z)

        # straight-through estimator
        quantized_st = z + (quantized - z).detach()

        # 4. commitment loss + codebook loss
        commit_loss = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z.detach())

        # 5. alignment loss (contrastive)
        flat_texts = [t for segs in seg_texts_nested for t in segs]
        text_emb = self.encode_text(flat_texts, device)          # (total_segs, 512)
        text_z = self.text_proj(text_emb)                        # (total_segs, coarse_dim)

        m_norm = F.normalize(quantized_st, dim=-1)
        t_norm = F.normalize(text_z, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100)
        logits = scale * m_norm @ t_norm.T                       # (total_segs, total_segs)
        labels = torch.arange(logits.shape[0], device=device)
        align_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

        total_loss = commit_loss + codebook_loss + align_loss

        return total_loss, commit_loss.item(), codebook_loss.item(), align_loss.item(), indices
