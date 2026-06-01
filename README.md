# 경쟁적 손실 융합: MoMask + Segmo

[MoMask](https://arxiv.org/abs/2312.00063) (CVPR 2024)를 기반으로, 세그먼트 단위 텍스트 조건화(Segmo)를 구현하고, 이어서 두 구조의 손실을 경쟁적으로 활용하는 새로운 연구를 진행합니다.

---

## 연구 개요

### Stage 1: Segmo 구현 (완료)

MoMask 위에 세그먼트 조건부 모션 생성을 구현했습니다. 각 모션 구간이 전체 텍스트 조건 외에 해당 구간만의 텍스트 설명으로 추가 조건화됩니다.

- 세그먼트별 텍스트 조건 토큰을 전역 조건 토큰과 함께 마스크 트랜스포머에 입력
- 정렬 손실(`Lalign`)을 통해 세그먼트 조건 벡터가 해당 모션 구간 벡터와 정렬되도록 유도
- **결과**: MoMask 베이스라인 대비 성능 저하 확인. 세그먼트 조건화가 생성 품질을 일관되게 향상시키지 못함.

### Stage 2: 경쟁적 손실 융합 (진행 중)

MoMask와 Segmo의 학습 목표를 **경쟁적 손실** 메커니즘으로 결합하는 새로운 연구 방향입니다.

**핵심 아이디어**: 매 학습 스텝마다 MoMask 손실과 Segmo 손실을 모두 계산하고, 단순 합산이 아닌 더 유효한 손실 신호(예: 더 낮거나 안정적이거나 그래디언트 크기가 큰 쪽)로 모델을 업데이트합니다.

이를 통해 두 구조가 경쟁하며, 어느 구조가 더 강한 학습 신호를 제공하느냐에 따라 훈련 신호가 적응적으로 라우팅됩니다.

**목표**:
- 고정된 Segmo 손실에서 발생한 성능 저하를 방지
- 세그먼트 조건화가 유효할 때만 선택적으로 활용하도록 유도
- 경쟁적 손실 선택이 다목적 모션 생성의 일반적 전략으로 기능하는지 검증

---

## 레포지토리 구조

```
.
├── models/
│   ├── mask_transformer/
│   │   └── transformer.py      # MaskTransformer + Segmo 확장
│   └── ...
├── run/                        # 학습 / 평가 / 생성 스크립트
├── etc/                        # 원본 MoMask README 및 자료
└── ...
```

---

## 기반 연구

- [MoMask (Guo et al., CVPR 2024)](https://arxiv.org/abs/2312.00063) — 3D 인간 모션의 생성적 마스크 모델링
- Segmo — 본 레포지토리에서 구현한 세그먼트 조건부 확장

```bibtex
@inproceedings{guo2024momask,
  title={Momask: Generative masked modeling of 3d human motions},
  author={Guo, Chuan and Mu, Yuxuan and Javed, Muhammad Gohar and Wang, Sen and Cheng, Li},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1900--1910},
  year={2024}
}
```
