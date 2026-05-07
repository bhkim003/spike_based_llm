
from modules.network import *
from modules.block import *

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import *

# modules 폴더에 새모듈.py 만들면
# modules/__init__py 파일에 form .새모듈 import * 하셈
# 그리고 새모듈.py에서 from modules.새모듈 import * 하셈


class _SpikeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qmin=-4.0, qmax=4.0):
        ctx.save_for_backward(x)
        ctx.qmin = qmin
        ctx.qmax = qmax
        return torch.round(torch.clamp(x, min=qmin, max=qmax))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = grad_output.clone()
        grad = torch.where((x >= ctx.qmin) & (x <= ctx.qmax), grad, torch.zeros_like(grad))
        return grad, None, None


class MultiSpike(nn.Module):
    # signed integer spike in [qmin, qmax]
    def __init__(self, qmin=-4.0, qmax=4.0):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax

    def forward(self, x):
        return _SpikeQuantSTE.apply(x, self.qmin, self.qmax)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # (dim,)

    def forward(self, x):
        # x: (..., dim)
        x_f = x.float()
        rstd = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x_f * rstd * self.weight
        return y.to(dtype=x.dtype)


class GatedMLP(nn.Module):
    def __init__(self, dim, hidden_mult=2):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.fc1 = nn.Linear(dim, hidden * 2)   # weight: (2*hidden, D)
        self.fc2 = nn.Linear(hidden, dim)        # weight: (D, hidden)

    def forward(self, x):
        # x: (B, L, D)
        x, gate = self.fc1(x).chunk(2, dim=-1)  # 각각 (B, L, hidden)
        return self.fc2(x * F.silu(gate))        # (B, L, D)


class SpikingMambaBlock(nn.Module):
    """
    Unified block:
    - pre-norm + spike-based Mamba-style mixer
    - optional GatedMLP residual branch

    논문 (Dao & Gu, 2024) Mamba2 수식 기준으로 구현:
      h_t = alpha_t * h_{t-1} + (dt_t * B_t) ⊗ x_t    (식 18)
      o_t = C_t @ h_t + D ⊙ x_t                        (식 19)
      y_t = Norm(concat(o)) ⊙ z                         (식 20)
    """
    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,       # d_inner = d_model * expand
        conv1d_kernel=4,
        headdim=64,
        A_init_range=(1, 16),
        dt_min=1e-3,
        dt_max=1e-1,
        dt_init_floor=1e-4,
        mlp_ratio=2,
        use_gated_mlp=False,
        spike_qmin=-4.0,
        spike_qmax=4.0,
        proj_bias=False,
        conv_bias=True,
        sgc_on=False,
    ):
        super().__init__()

        # ---- shape 기본값 ----
        self.d_model = d_model          # D
        self.d_state = d_state          # N
        self.expand = expand
        self.d_inner = d_model * expand # E = D * expand = H * P
        self.headdim = headdim          # P
        assert self.d_inner % headdim == 0, "d_model*expand must be divisible by headdim"
        self.nheads = self.d_inner // headdim   # H = E // P
        self.conv1d_kernel = conv1d_kernel      # K

        self.norm1 = RMSNorm(d_model)   # weight: (D,)

        # in_proj: D → 2E+2N+H  split order: [z(E), x(E), B(N), C(N), dt(H)]
        proj_out = self.d_inner + self.d_inner + d_state + d_state + self.nheads
        #          E              E              N          N          H
        self.in_proj = nn.Linear(d_model, proj_out,bias=proj_bias)      # weight: (2E+2N+H, D)
        self.out_proj = nn.Linear(self.d_inner, d_model,bias=proj_bias) # weight: (D, E)

        # conv1d: depthwise over [x(E), B(N), C(N)] 채널
        conv_dim = self.d_inner + 2 * d_state   # E + 2N
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,           # E + 2N
            out_channels=conv_dim,          # E + 2N
            kernel_size=conv1d_kernel,      # K
            padding=conv1d_kernel - 1,      # K-1  (causal padding, 뒤 K-1개 잘라냄)
            groups=conv_dim,                # depthwise
            bias=conv_bias,
        )
        # conv1d weight: (E+2N, 1, K),  bias: (E+2N,)

        # A_log: uniform [A_min, A_max] → log  →  (H,)
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads).uniform_(A_init_range[0], A_init_range[1])  # (H,)
        self.A_log = nn.Parameter(torch.log(A))  # (H,)  →  A = -exp(A_log) 항상 음수
        self.A_log._no_weight_decay = True

        # dt_bias: softplus^{-1}(dt_target)  →  (H,)
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)               # (H,)  log-uniform sample
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)      # (H,)
        self.dt_bias._no_weight_decay = True

        # D: skip connection
        # [레퍼런스 기본값]  scalar per head (D_has_hdim=False)
        self.D = nn.Parameter(torch.ones(self.nheads))           # (H,)
        # [논문 식 19 그대로]  D^(d) ∈ R^P, 채널마다 독립적인 skip 계수
        # self.D = nn.Parameter(torch.ones(self.nheads, self.headdim))  # (H, P)
        self.D._no_weight_decay = True

        self.mix_norm = RMSNorm(self.d_inner)   # weight: (E,)   (논문 식 20의 Norm)
        self.spike_in  = MultiSpike(spike_qmin, spike_qmax)
        self.spike_out = MultiSpike(spike_qmin, spike_qmax)

        # ---- Optional MLP branch ----
        self.use_gated_mlp = use_gated_mlp and (mlp_ratio is not None) and (mlp_ratio > 0)
        if self.use_gated_mlp:
            self.norm2 = RMSNorm(d_model)                        # weight: (D,)
            self.gated_mlp = GatedMLP(d_model, hidden_mult=mlp_ratio)
        else:
            self.norm2 = None
            self.gated_mlp = None

        self.sgc_on = sgc_on

    def forward(self, x):
        # x: (B, L, D)

        # ---- Mixer path ----
        u = self.norm1(x)           # (B, L, D)  prenorm
        B, L, _ = u.shape

        proj_smooth = None
        if self.sgc_on:
            proj_smooth = self.spike_in.qmax * torch.tanh(u)   # (B, L, D)
            proj_smooth = self.in_proj(proj_smooth)            # (B, L, 2E+2N+H)

        s = self.spike_in(u)        # (B, L, D)  spike quantize
        proj = self.in_proj(s)      # (B, L, 2E+2N+H)


        z, x_mix, Bv, Cv, dt = torch.split(
            proj,
            [self.d_inner, self.d_inner, self.d_state, self.d_state, self.nheads],
            dim=-1,
        )
        # z:     (B, L, E)   gating (식 20의 z)
        # x_mix: (B, L, E)   SSM input
        # Bv:    (B, L, N)   input matrix B (conv 전)
        # Cv:    (B, L, N)   output matrix C (conv 전)
        # dt:    (B, L, H)   time step raw (softplus 전)

        # depthwise causal conv1d: x_mix, Bv, Cv 채널 묶어서 처리
        xbc = torch.cat([x_mix, Bv, Cv], dim=-1)   # (B, L, E+2N)
        xbc = xbc.transpose(1, 2)                   # (B, E+2N, L)
        xbc = self.conv1d(xbc)[..., :L]             # (B, E+2N, L)  causal: 뒤 K-1개 버림
        xbc = F.silu(xbc).transpose(1, 2)           # (B, L, E+2N)

        x_mix, Bv, Cv = torch.split(xbc, [self.d_inner, self.d_state, self.d_state], dim=-1)
        # x_mix: (B, L, E)
        # Bv:    (B, L, N)
        # Cv:    (B, L, N)

        xh = x_mix.view(B, L, self.nheads, self.headdim)   # (B, L, H, P)
        dt = F.softplus(dt + self.dt_bias.view(1, 1, -1))  # (B, L, H)
        #                         dt_bias: (H,) → (1, 1, H)

        # ---- SSM scan  (논문 식 18, 19) ----
        H = self.nheads
        A = -torch.exp(self.A_log).to(dtype=xh.dtype)  # (H,)  항상 음수
        state = torch.zeros(B, H, self.headdim, self.d_state, device=xh.device, dtype=xh.dtype)
        #       state h_t:  (B, H, P, N)
        y = torch.zeros(B, L, H, self.headdim, device=xh.device, dtype=xh.dtype)
        #   output o_t:     (B, L, H, P)

        for t in range(L):
            # ── alpha_t = exp(A * dt_t)  ──────────────────────────────
            # A:       (H,)  →  (1, H)
            # dt[:,t]: (B, H)
            alpha = torch.exp(A.view(1, H) * dt[:, t])         # (B, H)
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)           # (B, H, 1, 1)

            # ── (dt_t * B_t) ⊗ x_t  ─────────────────────────────────
            # 논문 식 18: B에 dt를 곱한 뒤 x와 outer product
            # dt[:,t]:  (B, H)    → (B, H, 1)
            # Bv[:,t]:  (B, N)    → (B, 1, N)
            # dt*B:     (B, H, N) → unsqueeze(-2) → (B, H, 1, N)
            dtB_t = (dt[:, t].unsqueeze(-1) * Bv[:, t].unsqueeze(1)).unsqueeze(-2)
            #        (B, H, 1) * (B, 1, N) = (B, H, N) → (B, H, 1, N)

            # x_t: (B, H, P) → (B, H, P, 1)
            x_t = xh[:, t].unsqueeze(-1)                       # (B, H, P, 1)

            # ── state update:  h_t = alpha * h_{t-1} + (dt*B) ⊗ x  ──
            state = state * alpha + x_t * dtB_t
            # state * alpha:  (B, H, P, N) * (B, H, 1, 1) → (B, H, P, N)
            # x_t * dtB_t:    (B, H, P, 1) * (B, H, 1, N) → (B, H, P, N)  outer product

            # ── output:  o_t = C_t @ h_t  ────────────────────────────
            # c_t: (B, N) → (B, 1, 1, N)
            c_t = Cv[:, t].unsqueeze(1).unsqueeze(1)           # (B, 1, 1, N)
            y[:, t] = (state * c_t).sum(dim=-1)
        # xh:  (B, L, H, P)
        # [레퍼런스]  D: (H,) → broadcast over P
        # ── skip:  o_t += D ⊙ x_t  (논문 식 19)  ───────────────────
        # [논문 식 19]  D: (H, P) → 채널마다 독립 (D_has_hdim=True)
        # y = y + xh * self.D.unsqueeze(0).unsqueeze(0)        # (B, L, H, P)
        y = y + xh * self.D.view(1, 1, self.nheads, 1)         # (B, L, H, P)
        y = y.reshape(B, L, self.d_inner)   # (B, L, H*P) = (B, L, E)

        # ── 논문 식 20: Norm(concat(o)) ⊙ z  ─────────────────────────
        y = self.mix_norm(y * F.silu(z))    # (B, L, E) * (B, L, E) → (B, L, E)

        y_smooth = None
        if self.sgc_on:
            y_smooth = self.spike_out.qmax * torch.tanh(y) # (B, L, E)
            y_smooth = self.out_proj(y_smooth)            # (B, L, D)
        y = self.spike_out(y)               # (B, L, E)
        y = self.out_proj(y)                # (B, L, D)   (논문 식 21 Wout)
        

        x = x + y                           # (B, L, D)   residual

        # ---- Optional MLP path ----
        if self.use_gated_mlp:
            x = x + self.gated_mlp(self.norm2(x))  # (B, L, D)

        if self.sgc_on:
            return x, proj, proj_smooth, y, y_smooth
        else:
            return x, None, None, None, None

