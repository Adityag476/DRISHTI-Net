#!/usr/bin/env python3
"""DRISHTI-Net — reference architecture (build-locked, MASTER_SPEC v3.0).

FiLM-conditioned NAFNet U-Net for joint denoise + deblur + 2x super-resolution.
Decoder up-sampling contract: Conv1x1(c -> 2c) + PixelShuffle(2) -> exactly c/2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class LayerNorm2d(nn.Module):
    """LayerNorm across the channel dim (NAFNet style), NCHW in/out."""

    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(c, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class NAFBlockFiLM(nn.Module):
    """NAFBlock (SimpleGate + SCA) with FiLM conditioning on the spatial stream."""

    def __init__(self, c, z_dim=64, use_film=True):
        super().__init__()
        dw = c * 2
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, dw, 1)
        self.conv5 = nn.Conv2d(dw // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.use_film = use_film
        self.probe = False                                # diagnostics telemetry (train.py)
        if use_film:
            self.film = nn.Linear(z_dim, 2 * c)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x, z=None):
        y = self.conv3(self.sg(self.conv2(self.conv1(self.norm1(x)))))
        y = y * self.sca(y)
        if self.use_film and z is not None:
            g, b = self.film(z).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
            g, b = 0.5 * torch.tanh(g), 0.5 * torch.tanh(b)   # bounded modulation (§11 fix)
            if self.probe:
                gf, bf = g.detach().abs().flatten(), b.detach().abs().flatten()
                self._g_absmax = gf.max().item()
                self._b_absmax = bf.max().item()
                self._g_flat, self._b_flat = gf, bf       # probe steps only; ~B*c floats
            y = y * (1.0 + g) + b
        x = x + y * self.beta
        y = self.conv5(self.sg(self.conv4(self.norm2(x))))
        return x + y * self.gamma


class DegradationEncoder(nn.Module):
    """Implicit degradation encoder: input -> z (R^z_dim).
    Aux head (train-time only, synthetic batches) predicts
    [sigma_speckle, sigma_blur, sigma_noise, post_spk, post_noise (5 reg, aux_dim-5)
     | kernel 4-way logits | order logit] — the two post-downsample terms are the
    DOMINANT corruptions per the Day-0 audit (§10c) and must be supervised (R2-#1)."""

    def __init__(self, in_ch=2, z_dim=64, aux_dim=8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(64, z_dim)
        self.z_norm = nn.LayerNorm(z_dim)     # §11 postmortem: unbounded z + unbounded FiLM
        self.aux = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(True), nn.Linear(64, aux_dim))

    def forward(self, x):
        z = self.fc(self.body(x))
        z = self.z_norm(z)                    # standardized latent kills the z<->FiLM feedback loop
        return z, self.aux(z)


class DRISHTINet(nn.Module):
    """Width-32 NAFNet U-Net [2,2,4] / mid 4 / [2,2,2], FiLM(z) in every block,
    PixelShuffle x2 head, global bilinear residual. ~28 blocks, ~3.5M params."""

    def __init__(self, in_ch=2, out_ch=1, width=32, z_dim=64,
                 enc_blocks=(2, 2, 4), mid_blocks=4, dec_blocks=(2, 2, 2),
                 use_film=True, with_uncertainty=False, aux_dim=8):
        super().__init__()
        self.use_film = use_film
        self.probe = False                                # diagnostics telemetry (train.py)
        self.deg_enc = DegradationEncoder(in_ch, z_dim, aux_dim)
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)

        self.enc, self.downs = nn.ModuleList(), nn.ModuleList()
        for i, n in enumerate(enc_blocks):
            c = width * (2 ** i)
            blocks = nn.ModuleList([NAFBlockFiLM(c, z_dim, use_film) for _ in range(n)])
            for b in blocks:
                b._stage = f"enc{i + 1}"              # stage tag for by-depth telemetry
            self.enc.append(blocks)
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
        c_mid = width * (2 ** len(enc_blocks))
        self.mid = nn.ModuleList([NAFBlockFiLM(c_mid, z_dim, use_film) for _ in range(mid_blocks)])
        for b in self.mid:
            b._stage = "mid"

        chans = [width * (2 ** i) for i in range(len(enc_blocks) - 1, -1, -1)]  # 128,64,32
        self.dec, self.ups = nn.ModuleList(), nn.ModuleList()
        prev = c_mid
        for j, (n, c) in enumerate(zip(dec_blocks, chans)):
            self.ups.append(nn.Sequential(nn.Conv2d(prev, 2 * prev, 1, bias=False),
                                          nn.PixelShuffle(2)))                 # prev -> prev//2
            assert (2 * prev) // 4 == c, "decoder channel contract violated"
            blocks = nn.ModuleList([NAFBlockFiLM(c, z_dim, use_film) for _ in range(n)])
            for b in blocks:
                b._stage = f"dec{len(dec_blocks) - j}"   # dec3 pairs enc3, ... dec1 pairs enc1
            self.dec.append(blocks)
            prev = c

        self.head = nn.Sequential(nn.Conv2d(width, width * 4, 3, padding=1),
                                  nn.PixelShuffle(2),
                                  nn.Conv2d(width, out_ch, 3, padding=1))
        self.with_uncertainty = with_uncertainty
        if with_uncertainty:
            self.unc_head = nn.Sequential(nn.Conv2d(width, width * 4, 3, padding=1),
                                          nn.PixelShuffle(2),
                                          nn.Conv2d(width, 1, 3, padding=1), nn.Softplus())

    def forward(self, x, need_aux=False, need_unc=False):
        # degradation encoder only runs when FiLM is on or aux labels are requested
        if self.use_film or need_aux:
            z, aux = self.deg_enc(x)
            if self.probe and z is not None:
                zf = z.detach().abs()
                self._z_absmax = zf.max().item()
                self._z_absmean = zf.mean().item()
            if not self.use_film:
                z = None                            # explicit: FiLM disabled -> discard z
        else:
            z, aux = None, None
        f = self.intro(x)
        skips = []
        for blocks, down in zip(self.enc, self.downs):
            for blk in blocks:
                f = blk(f, z)
            skips.append(f)
            f = down(f)
        for blk in self.mid:
            f = blk(f, z)
        for blocks, up, skip in zip(self.dec, self.ups, reversed(skips)):
            f = up(f) + skip
            for blk in blocks:
                f = blk(f, z)
        base = F.interpolate(x[:, :1], scale_factor=2, mode="bilinear", align_corners=False)
        out = base + self.head(f)                      # global residual
        extras = []
        if need_aux:
            extras.append(aux)
        if need_unc and self.with_uncertainty:
            extras.append(self.unc_head(f))
        return (out, *extras) if extras else out


def build_from_config(cfg):
    return DRISHTINet(**{k: cfg[k] for k in
                         ("in_ch", "out_ch", "width", "z_dim", "use_film", "with_uncertainty", "aux_dim")
                         if k in cfg})


if __name__ == "__main__":
    m = DRISHTINet()
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n/1e6:.2f}M")
    for hw in (128, 256):
        x = torch.randn(1, 2, hw, hw, dtype=torch.float32)
        o, aux = m(x, need_aux=True)
        assert o.shape == (1, 1, 2 * hw, 2 * hw), o.shape
        print(f"in {hw}x{hw} -> out {o.shape[-1]}x{o.shape[-1]} OK; aux {tuple(aux.shape)}")
