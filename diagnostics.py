import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from Tokeniser import data, valid_w, valid_spectrum, valid_spectra, w, f, f_norm, patch_size, step_size, X, V

#------------------------------------------NOISE FLOOR (from flux errors)----------------------------------------------------#

# Same transform/filtering pipeline as 'f' in Tokeniser.py, applied to the error column
err = data['err'][np.ix_(valid_w, valid_spectrum)].T / (w**2)
err = err[valid_spectra]

# Propagate error through the arcsinh(f/scale) -> z-score transform via the chain rule:
# d(f_norm)/d(f) = 1 / (scale * sqrt(1+(f/scale)^2) * std(f_arcsinh))
scale = np.nanmedian(np.abs(f), axis=1, keepdims=True).clip(1e-30)
f_arcsinh = np.arcsinh(f / scale)
std_arcsinh = np.std(f_arcsinh, axis=1, keepdims=True)
deriv = 1.0 / (scale * np.sqrt(1 + (f / scale) ** 2) * std_arcsinh)
err_norm = err * deriv  # per-pixel noise std in f_norm units, (B, L)

err_patches = sliding_window_view(err_norm, patch_size, axis=1)[:, ::step_size]  # (B, T, P)
noise_var_per_pixel = np.nanmedian(err_patches ** 2)  # typical noise variance per raw pixel (z-score units), robust to outlier-error pixels
noise_floor_raw = noise_var_per_pixel * patch_size  # sum over P=4 raw-flux dims

print('=== Noise floor (from instrument flux errors) ===')
print(f'Avg per-pixel noise variance (z-score units): {noise_var_per_pixel:.4f}')
print(f'Noise floor contribution from 4 raw-flux dims: {noise_floor_raw:.4f}')
print(f'(out of patch_dim=6 total — mean/std dims add a bit more)')
print()

#------------------------------------------NEIGHBOR-INTERPOLATION BASELINE----------------------------------------------------#

rng = np.random.default_rng(0)
mask_ratio = 0.5
M = (rng.random(V.shape) < mask_ratio) & V

left, right = np.roll(X, 1, axis=1), np.roll(X, -1, axis=1)
left_v, right_v = np.roll(V, 1, axis=1), np.roll(V, -1, axis=1)

both, only_left, only_right = left_v & right_v, left_v & ~right_v, right_v & ~left_v

interp = np.zeros_like(X)
interp[both] = 0.5 * (left[both] + right[both])
interp[only_left] = left[only_left]
interp[only_right] = right[only_right]

err2 = ((X - interp) ** 2).sum(axis=-1)  # (B, T)
err2_masked = err2[M]

print('=== Neighbor-interpolation baseline (no model) ===')
print(f'Mean reconstruction loss: {err2_masked.mean():.4f}')
print(f'Compare to: predict-zero baseline ~6.0, current model val_loss ~0.7')
print()

#------------------------------------------OUTLIER CONTRIBUTION----------------------------------------------------#

sorted_err = np.sort(err2_masked)[::-1]
total = sorted_err.sum()
for pct in [0.001, 0.01, 0.05, 0.10]:
    n = max(1, int(pct * len(sorted_err)))
    frac = sorted_err[:n].sum() / total
    print(f'Top {pct:.1%} of masked tokens contribute {frac:.1%} of total interpolation loss')
