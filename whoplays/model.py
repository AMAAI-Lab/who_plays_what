"""The separator: MVSep Mega (53-stem BS-RoFormer, ZFTurbo, Apr 2026) as the audio trunk, told whom to keep by the clicked
player's video.

Mega's band split and its 12 blocks (a transformer along time, one across bands), loaded from its checkpoint; after each
time transformer, cross-attention from every band-time token to the clicked player's video tokens, its output layer
starting at zero so the untrained model is Mega's trunk. Mega's 53 per-stem mask heads are not used: one new head of the
same shape, from scratch, gives the clicked player's complex mask. Mega is stereo: the mono recording goes in on both
channels and the two outputs are averaged. STFT 2048 / hop 512 (Mega's).

Video tokens: the V-JEPA 2 grid (16 cells of 1024 per 2 frames) and the 48 body and hand keypoints (144 numbers per
frame), each projected with learned cell / kind embeddings, then 2 transformer layers. Every token carries its time in
10 ms steps (rotary), so the audio frames, the 12.5/s V-JEPA 2 and the 25/s keypoint tokens line up without resampling.

pretrained=False builds the same model without loading Mega's checkpoint (used to load a trained model).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from einops import pack, rearrange, unpack

from .bs_roformer.bs_roformer import BSRoformer, MaskEstimator


def rotary(pos, dim, base=10000.0):
    f = base ** (-torch.arange(0, dim, 2, device=pos.device) / dim)
    a = pos[:, None] * f
    return a.cos(), a.sin()


def rope(x, cos, sin):
    a, b = x.chunk(2, -1)
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


class Block(nn.Module):
    """Transformer layer: RMSNorm, rotary self-attention, SwiGLU."""

    def __init__(self, d, heads):
        super().__init__()
        self.heads, h = heads, 256 * round(8 * d / 3 / 256)
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.up, self.down = nn.Linear(d, 2 * h, bias=False), nn.Linear(h, d, bias=False)

    def forward(self, x, cos, sin):
        B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.heads, -1).transpose(1, 3).unbind(2)
        x = x + self.o(F.scaled_dot_product_attention(rope(q, cos, sin), rope(k, cos, sin), v).transpose(1, 2).reshape(B, N, d))
        a, b = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(a) * b)


class Cross(nn.Module):
    """Cross-attention from audio tokens to video tokens, rotary time positions on both."""

    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.nq, self.nk = nn.RMSNorm(d), nn.RMSNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d, bias=False), nn.Linear(d, 2 * d, bias=False), nn.Linear(d, d, bias=False)

    def forward(self, x, v, qpos, kpos):
        B, N, d = x.shape
        q = self.q(self.nq(x)).view(B, N, self.heads, -1).transpose(1, 2)
        k, val = self.kv(self.nk(v)).view(B, v.shape[1], 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        (cq, sq), (ck, sk) = rotary(qpos, d // self.heads), rotary(kpos, d // self.heads)
        a = F.scaled_dot_product_attention(rope(q, cq, sq), rope(k, ck, sk), val)
        return x + self.o(a.transpose(1, 2).reshape(B, N, d))


class AVMega(nn.Module):
    def __init__(self, cfg, pretrained=True, vjepa=1024, kp=144, cells=16, heads=8):
        super().__init__()
        conf = yaml.load(open(cfg["paths"]["mega_yaml"]), Loader=yaml.FullLoader)["model"]
        mega = BSRoformer(**conf)
        if pretrained:
            mega.load_state_dict(torch.load(cfg["paths"]["mega_ckpt"], map_location="cpu", weights_only=False), strict=True)
        self.d, self.heads, self.hop = conf["dim"], heads, conf["stft_hop_length"]
        self.band_split, self.layers, self.final_norm = mega.band_split, mega.layers, mega.final_norm
        self.stft_kwargs, self.win = mega.stft_kwargs, conf["stft_win_length"]
        self.mask = MaskEstimator(dim=self.d, dim_inputs=mega.band_split.dim_inputs, depth=conf["mask_estimator_depth"],
                                  mlp_expansion_factor=conf.get("mlp_expansion_factor", 4))
        self.trunk = [self.band_split, self.layers, self.final_norm]  # pretrained parts
        d = self.d
        self.cross = nn.ModuleList(Cross(d, heads) for _ in self.layers)
        for c in self.cross:
            nn.init.zeros_(c.o.weight)
        self.vj_in = nn.Sequential(nn.LayerNorm(vjepa), nn.Linear(vjepa, d))
        self.kp_in = nn.Sequential(nn.LayerNorm(kp), nn.Linear(kp, d))
        self.cell = nn.Parameter(torch.randn(cells, d) * 0.02)  # which of the 4 x 4 cells of the crop
        self.kind = nn.Parameter(torch.randn(2, d) * 0.02)  # V-JEPA 2 token or keypoint token
        self.vblocks = nn.ModuleList(Block(d, heads) for _ in range(2))

    def param_groups(self, lr):
        """The pretrained trunk learns at lr / 6, the new parts at lr."""
        trunk = {id(p) for m in self.trunk for p in m.parameters()}
        return [dict(params=[p for p in self.parameters() if id(p) in trunk], lr=lr / 6),
                dict(params=[p for p in self.parameters() if id(p) not in trunk], lr=lr)]

    def video_tokens(self, vj, kp):
        """vj (B, Tv, 16, 1024) at 12.5/s, kp (B, Tk, 144) at 25/s -> tokens (B, 16 Tv + Tk, d), times in 10 ms steps.
        Video frame f is at 40f ms; a V-JEPA 2 token covers frames 2j and 2j + 1."""
        x = torch.cat([(self.vj_in(vj) + self.cell + self.kind[0]).flatten(1, 2), self.kp_in(kp) + self.kind[1]], 1)
        dev = vj.device
        t = torch.cat([(8 * torch.arange(vj.shape[1], device=dev) + 2).repeat_interleave(vj.shape[2]),
                       4 * torch.arange(kp.shape[1], device=dev)]).float()
        cos, sin = rotary(t, self.d // self.heads)
        for b in self.vblocks:
            x = b(x, cos, sin)
        return x, t

    def forward(self, wave, vj, kp):
        """wave (B, L) recording at 44.1 kHz, vj and kp the clicked player's video -> (B, L) that player's sound."""
        B, L = wave.shape
        win = torch.hann_window(self.win, device=wave.device)
        X = torch.stft(wave.float()[:, None].expand(B, 2, L).reshape(2 * B, L), **self.stft_kwargs, window=win, return_complex=True)
        st = rearrange(torch.view_as_real(X), "(b s) f t c -> b (f s) t c", s=2)
        x = self.band_split(rearrange(st, "b f t c -> b t (f c)"))  # (B, T, K, d)
        T, K = x.shape[1], x.shape[2]
        v, vt = self.video_tokens(vj, kp)
        qt = (torch.arange(T, device=wave.device).float() * self.hop / 441).repeat(K)
        for i, (time_t, band_t) in enumerate(self.layers):
            x, ps = pack([rearrange(x, "b t f d -> b f t d")], "* t d")
            x = unpack(time_t(x), ps, "* t d")[0]  # (B, K, T, d)
            x = self.cross[i](x.reshape(B, K * T, self.d), v, qt, vt).view(B, K, T, self.d)
            x, ps = pack([rearrange(x, "b f t d -> b t f d")], "* f d")
            x = unpack(band_t(x), ps, "* f d")[0]
        m = rearrange(self.mask(self.final_norm(x)).float(), "b t (f c) -> b f t c", c=2)
        S = torch.view_as_complex(st.contiguous()) * torch.view_as_complex(m.contiguous())
        S = rearrange(S, "b (f s) t -> (b s) f t", s=2).clone()
        S[:, 0] = 0  # Mega's zero_dc
        return torch.istft(S, **self.stft_kwargs, window=win, length=L).view(B, 2, L).mean(1)


def load_model(cfg, ckpt):
    """A trained model from its checkpoint (Mega's weights are inside it, so they are not loaded again)."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = AVMega(cfg, pretrained=False).cuda().eval()
    model.load_state_dict(ck["model"])
    return model
