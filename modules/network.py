
from modules.network import *
from modules.block import *


class SpikingMamba(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        n_layer=24,
        d_state=128,
        expand=2,
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
        sgc_indices=None,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)  # weight: (vocab_size, D)

        # sgc_on=True 로 설정할 블록 인덱스 목록 (기본: 첫/중간/마지막)
        if sgc_indices is None:
            sgc_indices = [0, n_layer // 2, n_layer - 1]
        sgc_indices = set(sgc_indices)
        self.blocks = nn.ModuleList([
            SpikingMambaBlock(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                conv1d_kernel=conv1d_kernel,
                headdim=headdim,
                A_init_range=A_init_range,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
                mlp_ratio=mlp_ratio,
                use_gated_mlp=use_gated_mlp,
                spike_qmin=spike_qmin,
                spike_qmax=spike_qmax,
                proj_bias=proj_bias,
                conv_bias=conv_bias,
                sgc_on=(i in sgc_indices),
            )
            for i in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model)                             # weight: (D,)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # weight: (vocab_size, D)
        self.lm_head.weight = self.embed.weight                    # tie weights

    def forward(self, input_ids):
        # input_ids: (B, L)
        x = self.embed(input_ids)   # (B, L, D)

        sgc_outputs = []            # sgc_on 블록에서 나온 (proj, proj_smooth, y, y_smooth) 모음
        for blk in self.blocks:
            out = blk(x)            # 항상 5-tuple: (x, proj, proj_smooth, y, y_smooth)
            x = out[0]              # (B, L, D)  다음 블록으로 전달
            if blk.sgc_on:
                sgc_outputs.append(out[1:])   # (proj, proj_smooth, y, y_smooth)

        x = self.norm_f(x)          # (B, L, D)
        logits = self.lm_head(x)    # (B, L, vocab_size)

        # sgc_outputs: list of (proj, proj_smooth, y, y_smooth) for sgc_on blocks
        # 학습 시 L_Hidden 계산에 사용; 추론 시에는 빈 리스트
        return logits, sgc_outputs