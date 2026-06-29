"""SpecML — masked-patch spectral transformer, as a single LightningModule.

Following the LowResPT layout, SpecML *is* the LightningModule: it bundles the
architecture, the in-model preprocessing (normalisation + tokenisation +
wavelength positional encoding), the masked-patch training loop, and the
raw-data inference API (`embed`, `reconstruct`). Checkpoints are native Lightning
``.ckpt`` files; load with ``SpecML.load_from_checkpoint(...)`` (or the
``load_specml`` helper below).

The numerical preprocessing is identical to the original Tokeniser.py, just in
torch (verified on the real data).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR

from .modules import SpectralBlock


# -------------------------------------------------MASKING / LOSS-----------------------------------------------------------#

def apply_random_mask_batch(y_b: torch.Tensor, v_b: torch.Tensor,
                            mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly zero a fraction `mask_ratio` of the VALID tokens per spectrum.

    Returns the masked input and the boolean mask M (True = masked & supervised).
    """
    M = (torch.rand(y_b.shape[:2], device=y_b.device) < mask_ratio) & v_b
    x_b = y_b.clone()
    x_b[M] = 0.0
    return x_b, M


def mse_loss(Y, Yhat, M, w=None):
    """MSE over masked tokens. With per-token weights `w` (B,T), returns the
    WEIGHTED mean — normalised by Σw (not the token count) so the loss scale is
    invariant to the absolute weight magnitude (only relative weights matter)."""
    err = ((Y - Yhat) ** 2).sum(dim=-1)  # [B, T]  squared L2 norm over patch dim
    if w is None:
        return err[M].mean()  # mean over masked positions only
    wm = w[M]
    return (wm * err[M]).sum() / wm.sum().clamp(min=1e-8)


# -------------------------------------------------THE MODEL-----------------------------------------------------------#

class SpecML(pl.LightningModule):
    """Masked-patch spectral transformer + pre-training loop."""

    def __init__(
        self,
        # architecture
        embed_dim: int = 384,
        n_heads: int = 8,
        n_layers: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.0,
        # tokenisation
        patch_size: int = 4,
        overlap: int = 2,
        # masking
        mask_ratio: float = 0.5,
        # loss weighting: "none" = plain MSE; "invvar" = clamp inverse-variance
        # err weighting  w = 1/max(σ², σ²_min)  on the masked-token reconstruction
        # loss, where σ² = Σ_patch err_norm² is the per-token noise variance in
        # normalised-flux space. Down-weights noise-dominated tokens so the loss
        # tracks the learnable signal (and downstream) instead of the noise floor.
        loss_weighting: str = "none",
        err_weight_sigma_min: float = 0.5,
        # optimizer
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.95),
        # scheduler: linear warmup, then a single cosine decay to min_lr (no restarts)
        warmup_steps: int = 8000,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.patch_dim = patch_size + 2
        self.step = patch_size - overlap
        self.d_emb = embed_dim
        ff = int(ffn_ratio * embed_dim)

        # Patch embedding (named patch_embed so the public method can be `embed`).
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        nn.init.trunc_normal_(self.patch_embed.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)
        self.blocks = nn.ModuleList([SpectralBlock(embed_dim, n_heads, ff, dropout)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)  # stabilises training
        self.head = nn.Linear(embed_dim, self.patch_dim)  # decoder

    # ──────────────────────────────────────────────────────────────────────────
    # Core transformer
    # ──────────────────────────────────────────────────────────────────────────

    def _encode(self, X, V, P):
        x = self.patch_embed(X) + P
        for blk in self.blocks:
            x = blk(x, V)
        return self.norm(x)  # [B, T, D]

    def forward(self, X, V, P):
        """Low-level reconstruction from tokenised input [B,T,patch_dim]."""
        return self.head(self._encode(X, V, P))

    def pool(self, X, V, P):
        """Validity-weighted mean over tokens → pooled embedding [B, D]."""
        x = self._encode(X, V, P)
        mask = V.unsqueeze(-1).to(x.dtype)
        # clamp denom so a spectrum with zero valid tokens gives 0, not NaN
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    # ──────────────────────────────────────────────────────────────────────────
    # In-model preprocessing — identical logic to the original Tokeniser.py.
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def data_stretch(flux):
        """Per-spectrum normalisation — torch port of Tokeniser.normalise_flux.

        flux : (B, L) raw flux, already divided by λ² (f_lambda).

        scale by median(|flux|) → arcsinh → per-spectrum z-score. NaN-aware so a
        bad pixel can't poison the stats; computed over ALL pixels (per-pixel
        validity is applied later, at the token level), exactly like the original.

        Returns (flux_norm, scale, sd); `scale`/`sd` (B,1) are returned so the
        per-pixel error can be propagated into the same space (see err_stretch).
        """
        scale = torch.nanmedian(flux.abs(), dim=1, keepdim=True).values.clamp(min=1e-30)
        f_arcsinh = torch.arcsinh(flux / scale)
        mu = torch.nanmean(f_arcsinh, dim=1, keepdim=True)
        var = torch.nanmean((f_arcsinh - mu) ** 2, dim=1, keepdim=True)  # population var
        sd = var.sqrt()
        return (f_arcsinh - mu) / sd, scale, sd

    @staticmethod
    def err_stretch(flux, err, scale, sd):
        """Propagate per-pixel error into normalised-flux space (B, L).

        Chain rule through arcsinh(f/scale) then the per-spectrum z-score:
            d(f_norm)/d(f) = 1 / (scale · √(1+(f/scale)²) · sd)
        so err_norm = err · that. Non-finite err (e.g. inf bad-pixel flag) maps to
        inf, giving that token ~0 weight downstream.
        """
        return err / (scale * torch.sqrt(1.0 + (flux / scale) ** 2) * sd)

    def token_var(self, err_norm):
        """Per-token noise variance σ² = Σ_patch err_norm²  → (B, T)."""
        P, S = self.hparams.patch_size, self.step
        pe = err_norm.unfold(1, P, S)            # (B, T, P)
        return (pe ** 2).sum(dim=-1)

    def tokenize(self, flux_norm, valid_mask):
        """Patchify normalised flux into (X, V) — torch port of Tokeniser.tokenize.

        Returns X (B, T, patch_size+2) = [μ_patch, σ_patch, patch_values] with
        fully-invalid tokens zeroed, and V (B, T) bool (True iff every pixel in
        the patch is valid).
        """
        P, S = self.hparams.patch_size, self.step
        patches = flux_norm.unfold(1, P, S)              # (B, T, P)
        valid_p = valid_mask.bool().unfold(1, P, S)      # (B, T, P)
        V = valid_p.all(dim=-1)                          # (B, T) all-valid token

        mean = patches.mean(dim=-1, keepdim=True)
        std = patches.std(dim=-1, unbiased=False, keepdim=True)  # population std
        X = torch.cat([mean, std, patches], dim=-1)      # (B, T, P+2)
        X = X * V.unsqueeze(-1).to(X.dtype)              # zero invalid tokens
        return X, V

    def token_wavelength(self, wavelength):
        """Mean wavelength per patch — (B, T)."""
        P, S = self.hparams.patch_size, self.step
        return wavelength.unfold(1, P, S).mean(dim=-1)

    def positional_encoding(self, wave_token):
        """Sinusoidal wavelength encoding — torch port of the Tokeniser P block.

        Phase reaches ~5e4 for the highest-frequency channels, where sin/cos are
        very sensitive to rounding; evaluated in float64 (as the original numpy
        did) and cast back, so the encoding is stable. Returns (B, T, d_emb).
        """
        d = self.d_emb
        half = d // 2
        omegas = 10000.0 ** (
            -2.0 * torch.arange(half, device=wave_token.device, dtype=torch.float64) / d
        )
        product = (wave_token.to(torch.float64) * 1e4).unsqueeze(-1) * omegas  # (B, T, half)
        P_enc = torch.empty(*wave_token.shape, d, device=wave_token.device, dtype=torch.float64)
        P_enc[..., 0::2] = torch.sin(product)
        P_enc[..., 1::2] = torch.cos(product)
        return P_enc.to(torch.float32)

    def preprocess(self, flux, wavelength, valid_mask):
        """Raw flux → (X, V, P_enc), the full input the transformer consumes."""
        flux_norm, _, _ = self.data_stretch(flux)
        X, V = self.tokenize(flux_norm, valid_mask)
        P_enc = self.positional_encoding(self.token_wavelength(wavelength))
        return X, V, P_enc

    # ──────────────────────────────────────────────────────────────────────────
    # Raw-data inference API
    # ──────────────────────────────────────────────────────────────────────────

    def embed(self, flux, wavelength, valid_mask) -> dict:
        """Encode raw spectra into embeddings — LowResPT-style dict API.

        Runs the full encoder pipeline (preprocess → encode) from RAW input, so
        downstream code never re-implements normalisation/tokenisation. Not
        wrapped in no_grad — the caller controls the grad context (frozen probe
        vs. fine-tuning).

        Args:
            flux, wavelength, valid_mask: (B, L).

        Returns dict:
            pooled:           (B, D)    validity-weighted mean over tokens
                                        (the global spectrum embedding)
            tokens:           (B, T, D) per-token encoder outputs
            token_valid_mask: (B, T)    bool, True = valid token
        """
        X, V, P_enc = self.preprocess(flux, wavelength, valid_mask)
        tokens = self._encode(X, V, P_enc)                       # (B, T, D)
        m = V.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # (B, D)
        return {"pooled": pooled, "tokens": tokens, "token_valid_mask": V}

    @torch.no_grad()
    def reconstruct(self, flux, wavelength, valid_mask, masked=False,
                    mask_ratio=0.5, generator=None) -> dict:
        """Reconstruct patches from raw flux + wavelength + valid mask.

        masked=False : all tokens visible; one forward pass reconstructs every
                       patch (the regime val_loss / recon notebooks evaluate).
        masked=True  : randomly hide `mask_ratio` of the valid tokens (MAE
                       regime); `mask` flags the hidden tokens.

        Returns dict:
            recon:            (B, T, P+2) reconstructed [μ, σ, patch values]
            target:           (B, T, P+2) the tokenised input (= reconstruction target)
            valid:            (B, T) bool token validity
            mask:             (B, T) bool tokens that were hidden (== valid if not masked)
            flux_norm:        (B, L) per-spectrum normalised input flux
            wave_token:       (B, T) mean wavelength per token (for mapping back to λ)
        """
        flux_norm, _, _ = self.data_stretch(flux)
        X, V = self.tokenize(flux_norm, valid_mask)
        P_enc = self.positional_encoding(self.token_wavelength(wavelength))
        target = X.clone()
        if masked:
            coin = torch.rand(X.shape[:2], device=X.device, generator=generator)
            mask = (coin < mask_ratio) & V
            x_in = X.clone()
            x_in[mask] = 0.0
        else:
            x_in = X
            mask = V
        recon = self.forward(x_in, V, P_enc)
        return {"recon": recon, "target": target, "valid": V, "mask": mask,
                "flux_norm": flux_norm, "wave_token": self.token_wavelength(wavelength)}

    # ──────────────────────────────────────────────────────────────────────────
    # Training / validation
    # ──────────────────────────────────────────────────────────────────────────

    def _token_weights(self, batch, scale, sd):
        """Per-token inverse-variance weights w = 1/max(σ², σ²_min), or None.

        σ² = Σ_patch err_norm² is the token noise variance in normalised-flux
        space; clamping at σ²_min caps the weight of the best-measured tokens so
        a handful don't dominate the gradient (see the err-weighting study)."""
        if self.hparams.loss_weighting != "invvar" or "err" not in batch:
            return None
        err_norm = self.err_stretch(batch["flux"], batch["err"], scale, sd)
        sig2 = self.token_var(err_norm)
        sig2 = torch.nan_to_num(sig2, nan=1e6, posinf=1e6)
        return 1.0 / torch.clamp(sig2, min=self.hparams.err_weight_sigma_min)

    def _step(self, batch, weighted):
        flux, wav, vm = batch["flux"], batch["wavelength"], batch["valid_mask"]
        flux_norm, scale, sd = self.data_stretch(flux)
        X, V = self.tokenize(flux_norm, vm)
        P = self.positional_encoding(self.token_wavelength(wav))
        x_b, m_b = apply_random_mask_batch(X, V, self.hparams.mask_ratio)
        Yhat = self.forward(x_b, V, P)
        w = self._token_weights(batch, scale, sd) if weighted else None
        return mse_loss(X, Yhat, m_b, w)

    def training_step(self, batch, batch_idx):
        # Train with the err-weighting (if enabled).
        loss = self._step(batch, weighted=True)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("lr", lr, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # val_loss stays UNWEIGHTED — an unbiased, history-comparable physical
        # metric. Real model quality is judged by the downstream σ_NMAD callback.
        loss = self._step(batch, weighted=False)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Same recipe as LowResPT: linear warmup then a SINGLE cosine decay over
        # the whole run down to min_lr (no SGDR restarts). Total steps come from
        # the trainer so the cosine reaches its floor exactly at the end.
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=tuple(self.hparams.betas),
        )

        # Total steps for the cosine: normally from the trainer, but a caller can
        # set `model._total_steps_override` (e.g. Optuna, where a manual
        # trainer.fit makes estimated_stepping_batches unsafe at this point).
        total_steps = getattr(self, "_total_steps_override", None)
        if total_steps is None:
            total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_steps
        min_lr_ratio = self.hparams.min_lr / self.hparams.lr

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = LambdaLR(opt, lr_lambda)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}


# ─────────────────────────────────────────────────────────────────────────────
# Convenience loader (LowResPT-style: native Lightning checkpoints)
# ─────────────────────────────────────────────────────────────────────────────

def load_specml(path, device="cpu"):
    """Load a SpecML ``.ckpt`` and return the model in eval mode.

    Hyper-parameters are restored from the checkpoint; read them via
    ``model.hparams``. Loaded with ``weights_only=False`` because Lightning
    stores hparams (incl. numpy scalars) in the checkpoint, which PyTorch 2.6's
    default ``weights_only=True`` refuses — only load checkpoints you trust.
    """
    import inspect
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Keep only real __init__ args (Lightning/CLI may inject extras like
    # '_instantiator' into the saved hyper_parameters).
    valid = set(inspect.signature(SpecML.__init__).parameters) - {"self"}
    hparams = {k: v for k, v in ckpt.get("hyper_parameters", {}).items() if k in valid}
    model = SpecML(**hparams)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()
