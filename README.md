# Segmo

MoMask(CVPR 2024) 기반으로 SegMo(arxiv 2512.21237) 논문의 아이디어를 구현한 프로젝트.

전체 텍스트 캡션뿐만 아니라 **구간별로 분할된 텍스트(segmented caption)** 를 MaskTransformer에 함께 입력하고, 모션 구간과 텍스트 구간을 매칭하여 세밀한 생성 제어와 loss 가중치 조정을 목표로 한다.

---

## 기반 코드

- **MoMask** (CVPR 2024): https://arxiv.org/abs/2312.00063
- **SegMo** (참고 논문): https://arxiv.org/pdf/2512.21237

---

## 폴더 구조

```
Segmo/
├── data/
│   └── t2m_dataset.py          # Dataset 및 DataLoader 정의
├── models/
│   ├── mask_transformer/
│   │   ├── transformer.py      # MaskTransformer 모델 핵심
│   │   ├── transformer_trainer.py
│   │   └── tools.py
│   └── vq/                     # VQ-VAE 관련 모델
├── run/                        # 학습/평가/생성 실행 스크립트
├── options/                    # 학습 옵션 정의
├── utils/                      # 평가 지표, 유틸리티
├── motion_loaders/             # 생성된 모션 로딩
├── common/                     # 쿼터니언 등 공통 수학 모듈
├── visualization/              # 모션 시각화
├── glove/                      # GloVe 워드 임베딩
├── checkpoints/                # 저장된 모델 가중치
└── etc/                        # 환경 설정, 라이선스, 원본 README
```

---

## 데이터

- **HumanML3D** 기반 학습
- Segmented Caption 경로: `/data4/local_datasets/HumanML3D/SegmentedCaption/`
  - **JSONL** (`train.jsonl`): 전체 train 데이터를 한 파일에 저장. 모션 ID 기준으로 접근
    ```json
    {"id": "000001", "split": "train", "captions": [
        {"caption": "...", "segments": ["t1", "t2", "t3"], "n_segments": 3},
        {"caption": "...", "segments": ["t1", "t2"], "n_segments": 2}
    ]}
    ```
  - **개별 txt** (`{motionID}_{i}.txt`): caption별 segments를 HumanML3D 포맷으로 저장 (**1-indexed**, i=1부터 시작)
    - `000001_1.txt` → 첫번째 caption의 segments
    - `000001_2.txt` → 두번째 caption의 segments
  - **구현에서는 JSONL 사용** (txt는 참고용)
    - `seg_dict[motionID][line_idx]` 로 접근 (`line_idx`는 0-indexed)
    - txt 파일 직접 접근 시 주의: `line_idx=0` → `{id}_1.txt` (1-indexed offset 필요)

---

## 구현 계획

### Stage 1: MaskTransformer에 Segmented Caption 입력 추가

**현재 구조:**
```
[MASK된 모션 토큰들] + [전체 텍스트 condition 토큰 1개]
```

**변경 구조:**
```
[MASK된 모션 토큰들] + [전체 텍스트 토큰] + [t1 토큰] + [t2 토큰] + ... + [tN 토큰]
```

**구현 내용:**

1. **`data/t2m_dataset.py`**
   - `__init__`: `seg_dir` 파라미터 추가, init 시 `train.jsonl` 한 번만 로드해 `seg_dict` 구성
   - `__init__` 루프: `line_idx` 추적, `seg_captions` (segments 리스트 또는 None) 를 `data_dict`에 함께 저장
     - `f_tag == 0.0` 캡션: `data_dict[name]['seg_captions']` 리스트에 append
     - `f_tag != 0.0` 캡션: `data_dict[new_name]['seg_captions'] = [seg_captions]`
     - segments가 1개이면 None 저장 (분할 안 된 것으로 간주)
   - `__getitem__`: `random.choice` → `random.randint`으로 index 추적, `seg_captions` 함께 반환
   - `train_t2m_transformer.py`의 DataLoader에 `collate_fn` 추가 (seg_captions는 list 그대로 유지)

2. **`models/mask_transformer/transformer.py`**
   - `trans_forward()`: `cond` 단일 벡터 → 복수 condition 벡터 처리로 확장
   - 각 segment 텍스트를 `encode_text()`로 CLIP 임베딩
   - 동일한 `cond_emb` (Linear layer) 를 공유하여 각 벡터를 latent_dim으로 projection
   - condition 토큰들을 sequence 앞에 개별 토큰으로 추가:
     ```python
     # (1 + N_seg, b, latent_dim) → sequence 앞에 prepend
     cond_tokens = torch.stack([full_token, t1_token, t2_token, ...], dim=0)
     xseq = torch.cat([cond_tokens, x], dim=0)
     ```
   - `padding_mask`도 condition 토큰 수만큼 앞에 확장

**설계 결정:**
- Segment condition 토큰에는 별도 positional encoding 추가하지 않음 (논문 따름)
- 같은 `cond_emb` layer를 전체 텍스트 및 모든 segment에 공유 사용
- Transformer self-attention이 모션 토큰 위치에 따라 적절한 segment 토큰에 attention하도록 학습에 맡김

**padding mask 처리:**
- 배치 내 샘플마다 segment 수가 다를 때, 최대 segment 수에 맞춰 패딩
- segment가 없는 자리는 0벡터로 채우고, `seg_valid_masks`를 통해 해당 위치를 `padding_mask=True`로 마스킹
- transformer가 유효하지 않은 segment 토큰에 attention하지 않도록 처리

---

### Stage 2: Motion Segment Aggregation

- 모션 프레임을 segment 수(N)로 **균등 분할**
  - 예: 모션 길이 90프레임, segment 3개 → [0:30], [30:60], [60:90]
- 각 구간의 모션 토큰과 대응하는 segment 텍스트를 매핑
- VQ-VAE로 인코딩된 모션 코드 레벨에서 구간 매핑 처리

---

### Stage 3: Segmented Loss 구현

- 모션 구간별로 대응하는 segment caption의 의미적 정합성에 따라 **loss 가중치 조정**
- 전체 텍스트 condition loss + segment별 추가 loss term
- SegMo 논문의 loss 설계 참조

---

## 현재 진행 상황

- [x] MoMask 기반 코드 정리 및 폴더 구조 재구성
- [x] Stage 1: Dataset segmented caption 로딩 구현
- [x] Stage 1: MaskTransformer / ResidualTransformer condition 토큰 확장
- [x] Stage 1: invalid segment 위치 padding mask 처리
- [ ] Stage 2: Motion segment aggregation
- [ ] Stage 3: Segmented loss 구현
