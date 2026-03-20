import torch
import torch.nn.functional as F

from utils import (
    load_wav_mono, 
    audio_to_latents,
    n_q_to_bandwidth,
    encode_audio_to_tokens,
    build_LOOKUP_via_layer_decode,
    latents128_to_audio,
    tokens_to_summary_latents,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_pool_weights(
    weights: list[float] | torch.Tensor | None,
    num_pools: int,
    num_frames: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if weights is None:
        return torch.full((num_pools, 1, 1), 1.0 / num_pools, device=device, dtype=dtype)

    weight_tensor = torch.as_tensor(weights, device=device, dtype=dtype)

    if weight_tensor.ndim == 1:
        if weight_tensor.shape[0] != num_pools:
            raise ValueError(f"Expected {num_pools} pool weights, got {weight_tensor.shape[0]}")
        if torch.any(weight_tensor < 0):
            raise ValueError("Pool weights must be non-negative")
        weight_sum = weight_tensor.sum().clamp_min(torch.finfo(dtype).eps)
        return (weight_tensor / weight_sum).view(num_pools, 1, 1)

    if weight_tensor.ndim == 2:
        if weight_tensor.shape != (num_frames, num_pools):
            raise ValueError(
                f"Expected per-frame weights with shape ({num_frames}, {num_pools}), "
                f"got {tuple(weight_tensor.shape)}"
            )
        if torch.any(weight_tensor < 0):
            raise ValueError("Pool weights must be non-negative")
        weight_sum = weight_tensor.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
        return (weight_tensor / weight_sum).transpose(0, 1).unsqueeze(-1)

    raise ValueError("Pool weights must be None, a 1D tensor/list, or a 2D per-frame tensor")


def _two_pool_weights(morph: float | torch.Tensor) -> float | torch.Tensor:
    morph_tensor = torch.as_tensor(morph, dtype=torch.float32)

    if morph_tensor.ndim == 0:
        return torch.stack((1 - morph_tensor, morph_tensor))

    if morph_tensor.ndim == 1:
        return torch.stack((1 - morph_tensor, morph_tensor), dim=-1)

    if morph_tensor.ndim == 2 and morph_tensor.shape[1] == 1:
        squeezed = morph_tensor.squeeze(1)
        return torch.stack((1 - squeezed, squeezed), dim=-1)

    raise ValueError("morph must be a scalar, shape (T,), or shape (T, 1)")


SAMPLE_RATE = 24000


def _augment_pool_audio(
    pool_audio: torch.Tensor,
    augmentations: list,
    sample_rate: int = SAMPLE_RATE,
) -> list[torch.Tensor]:
    """
    Apply each pedalboard effect chain to pool audio.

    Args:
        pool_audio:     Tensor of shape (1, T) — mono waveform.
        augmentations:  List of pedalboard.Pedalboard instances.  Each entry
                        produces one augmented copy of the pool audio.
        sample_rate:    Audio sample rate (default 24 kHz for Encodec).
    Returns:
        List of augmented audio tensors, each of shape (1, T').
    """
    audio_np = pool_audio.numpy()          # (1, T)  float32
    augmented = []
    for board in augmentations:
        processed = board(audio_np, sample_rate)
        augmented.append(torch.from_numpy(processed))
    return augmented


def encode_target(model, target_file: str, n_q: int) -> torch.Tensor:
    """Encode target audio file to continuous latents (pre-RVQ)."""
    target_audio = load_wav_mono(target_file)
    target_latents, _, _ = audio_to_latents(
        target_audio, model, DEVICE, n_q_to_bandwidth(n_q)
    )
    return target_latents


def encode_pool(
    model,
    pool_file: str,
    n_q: int,
    pool_augmentations: list | None = None,
) -> torch.Tensor:
    """
    Encode pool audio to token-derived summary latents.

    When *pool_augmentations* is provided, each augmentation chain is applied
    to the pool audio and the resulting latents are concatenated with the
    original, expanding the pool's codebook coverage.
    """
    pool_audio = load_wav_mono(pool_file)

    pool_variants = [pool_audio]
    if pool_augmentations:
        pool_variants.extend(
            _augment_pool_audio(pool_audio, pool_augmentations)
        )

    bandwidth = n_q_to_bandwidth(n_q)
    K = int(getattr(model.config, "codebook_size", 1024))

    first_tokens, _ = encode_audio_to_tokens(
        pool_variants[0], model, DEVICE, bandwidth
    )
    n_q_data = int(first_tokens.shape[1])
    LOOKUP_QKD = build_LOOKUP_via_layer_decode(
        model, n_q=n_q_data, K=K, device=DEVICE
    )

    all_pool_latents = [tokens_to_summary_latents(first_tokens, LOOKUP_QKD)]

    for variant in pool_variants[1:]:
        tokens, _ = encode_audio_to_tokens(variant, model, DEVICE, bandwidth)
        all_pool_latents.append(tokens_to_summary_latents(tokens, LOOKUP_QKD))

    return torch.cat(all_pool_latents, dim=0)


def extract_latents(
    model,
    target_file: str,
    pool_file: str,
    n_q: int,
    pool_augmentations: list | None = None,
):
    """
    Encode target and pool audio to latent representations.

    Convenience wrapper around encode_target / encode_pool.
    """
    target_latents = encode_target(model, target_file, n_q)
    pool_latents = encode_pool(model, pool_file, n_q, pool_augmentations)
    return target_latents, pool_latents


def decode_latents_to_audio(model, hybrid):
    # DECODE hybrid latents → audio
    hybrid_audio = latents128_to_audio(model, hybrid, DEVICE).cpu()
    print("Hybrid audio shape:", hybrid_audio.shape)
    return hybrid_audio.numpy()
    # display(Audio(hybrid_audio.numpy(), rate=24000))


# Cosine Similarity
def tokui_style_transfer_cosine(
    model,
    target_file: str,
    pool_file: str,
    n_q: int,
    pool_augmentations: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Style transfer with cosine similarity.
        target_file:  Path to the target audio file whose temporal structure
                      is preserved in the output.
        pool_file:    Path to the source/pool audio file whose timbral
                      characteristics are transferred to the output.
        n_q:          Number of RVQ quantizers used during encoding. Higher
                      values retain more detail but increase codebook size.
        pool_augmentations:  Optional list of pedalboard.Pedalboard effect
                      chains applied to the pool audio before encoding.
    """
    target_latents, pool_latents = extract_latents(
        model, target_file, pool_file, n_q,
        pool_augmentations=pool_augmentations,
    )

    t_norm = F.normalize(target_latents, dim=-1)
    p_norm = F.normalize(pool_latents,   dim=-1)

    sim = torch.mm(t_norm, p_norm.T)  # (T_target, T_pool)

    match_indices = sim.argmax(dim=-1)

    hybrid_latents = pool_latents[match_indices]
    return hybrid_latents, match_indices


def _match_windows(
    target_latents: torch.Tensor,
    pool_latents: torch.Tensor,
    window_size: int = 3,
    hop: int = 1,
    grain_size: int = 1,
    stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Core windowed cosine-similarity matching on pre-computed latents.

    Args:
        window_size: Frames concatenated for the similarity query vector.
        hop: Stride between consecutive pool candidate windows.
        grain_size: Number of consecutive pool frames copied per match.
                    - 1: single-frame grains (default, current behaviour).
                    - >1: multi-frame grains for smoother harmonic content.
        stride: Step between target query positions.
                    - stride == grain_size: grains tile without overlap.
                    - stride < grain_size: overlapping grains, crossfaded
                      with a triangular window.
                    - stride > grain_size: gaps filled by stretching the
                      nearest grain boundary.
    Returns:
        hybrid_latents of shape (T_target, D) and frame_match_indices.
    """
    T_target = target_latents.shape[0]
    T_pool   = pool_latents.shape[0]
    D        = target_latents.shape[1]
    pad      = window_size // 2

    target_padded = F.pad(target_latents.T.unsqueeze(0), (pad, pad), mode='reflect').squeeze(0).T
    pool_padded   = F.pad(pool_latents.T.unsqueeze(0),   (pad, pad), mode='reflect').squeeze(0).T

    # Query positions in the target: one every `stride` frames
    query_starts = list(range(0, T_target, stride))

    # Build target windows at query positions only
    target_windows = torch.stack([
        target_padded[t : t + window_size].reshape(-1)
        for t in query_starts
    ])  # (n_queries, window_size * D)

    # Pool candidate windows (unchanged from before)
    pool_starts = list(range(0, T_pool, hop))
    pool_windows = torch.stack([
        pool_padded[s : s + window_size].reshape(-1)
        for s in pool_starts
    ])  # (n_pool_w, window_size * D)

    t_norm = F.normalize(target_windows, dim=-1)
    p_norm = F.normalize(pool_windows,   dim=-1)

    sim = torch.mm(t_norm, p_norm.T)  # (n_queries, n_pool_w)
    window_match_indices = sim.argmax(dim=-1)

    # Convert window indices back to pool frame indices
    frame_match_indices = (window_match_indices * hop).clamp(0, T_pool - 1)

    # --- grain_size == 1 and stride == 1: fast path (original behaviour) ---
    if grain_size == 1 and stride == 1:
        hybrid_latents = pool_latents[frame_match_indices]
        return hybrid_latents, frame_match_indices

    # --- multi-frame grain assembly via overlap-add ---
    output  = torch.zeros(T_target, D, device=pool_latents.device, dtype=pool_latents.dtype)
    weights = torch.zeros(T_target, 1, device=pool_latents.device, dtype=pool_latents.dtype)

    # Triangular fade window for crossfading overlapping grains
    grain_window = torch.bartlett_window(grain_size, device=pool_latents.device, dtype=pool_latents.dtype)
    if grain_size == 1:
        grain_window = torch.ones(1, device=pool_latents.device, dtype=pool_latents.dtype)
    grain_window = grain_window.unsqueeze(-1)  # (grain_size, 1)

    # Expand frame_match_indices back to per-query for caller introspection
    full_match_indices = torch.full((T_target,), -1, dtype=torch.long,
                                    device=pool_latents.device)

    for qi, qs in enumerate(query_starts):
        src_start = int(frame_match_indices[qi].item())
        src_end   = min(src_start + grain_size, T_pool)
        actual_gs = src_end - src_start

        dst_end = min(qs + actual_gs, T_target)
        length  = dst_end - qs

        grain = pool_latents[src_start : src_start + length]  # (length, D)
        w     = grain_window[:length]                         # (length, 1)

        output[qs:dst_end]  += grain * w
        weights[qs:dst_end] += w

        full_match_indices[qs:dst_end] = torch.arange(
            src_start, src_start + length, device=pool_latents.device
        )

    # Normalise overlapping regions
    weights = weights.clamp_min(1e-8)
    hybrid_latents = output / weights

    return hybrid_latents, full_match_indices


def tokui_style_transfer_window(
    model,
    target_file: str,
    pool_file: str,
    n_q: int = 8,
    window_size: int = 3,
    hop: int = 1,
    grain_size: int = 1,
    stride: int = 1,
    pool_augmentations: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-frame context matching with hop-strided pool windows.
        window_size:  Number of consecutive latent frames concatenated to form
                      each window vector. Larger values capture broader local
                      context but makes matching less sensitive to fine
                      frame-level detail.Should be odd so padding is symmetric.
        hop:          Stride between consecutive pool windows. 
                      - hop=1: dense pool coverage, every frame is a candidate.
                      - hop>1: sparser pool windows, fewer candidates, coarser
                        grain jumps. Useful for forcing diversity in selection
                        or reducing memory when the pool is very large.
        grain_size:   Number of consecutive pool frames copied per match.
                      - 1: single-frame grains (percussive sources).
                      - 3-5: multi-frame grains (harmonic/sustained sources).
        stride:       Step between consecutive target query positions.
                      - stride == grain_size: grains tile without overlap.
                      - stride < grain_size: overlapping grains, crossfaded.
                      - stride == 1: maximum coverage (default).
        pool_augmentations:  Optional list of pedalboard.Pedalboard effect
                      chains applied to the pool audio before encoding.
    """
    target_latents, pool_latents = extract_latents(
        model, target_file, pool_file, n_q,
        pool_augmentations=pool_augmentations,
    )
    return _match_windows(target_latents, pool_latents, window_size, hop,
                          grain_size, stride)


# Interpolation
def tokui_style_transfer_blend(
    model,
    target_file: str,
    pool_file: str,
    n_q: int = 8,
    alpha: float = 1.0,
    window_size: int = 3,
    hop: int = 1,
    grain_size: int = 1,
    stride: int = 1,
    pool_augmentations: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Style transfer with α-blending between matched pool latents and target latents.
    """
    target_latents, pool_latents = extract_latents(
        model, target_file, pool_file, n_q,
        pool_augmentations=pool_augmentations,
    )

    matched_latents, match_indices = _match_windows(
        target_latents, pool_latents, window_size, hop,
        grain_size, stride,
    )

    # α=1 → pure pool, α=0 → pure target
    hybrid_latents = alpha * matched_latents + (1 - alpha) * target_latents

    return hybrid_latents, match_indices


def tokui_style_transfer_multi_pool_morph(
    model,
    target_file: str,
    pool_files: list[str],
    n_q: int = 8,
    weights: list[float] | torch.Tensor | None = None,
    window_size: int = 3,
    hop: int = 1,
    grain_size: int = 1,
    stride: int = 1,
    pool_augmentations: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Style transfer with weighted morphing across an arbitrary number of source pools.

    Each pool is matched independently against the same target, then the matched
    latent streams are combined with either fixed weights or per-frame weights.

    Args:
        target_file: Path to the target audio file whose temporal structure is
            preserved in the output.
        pool_files: List of source pools to match against.
        n_q: Number of RVQ quantizers used during encoding.
        weights: Pool mixing weights.
            - None: equal weighting across all pools.
            - shape (N,): fixed weight per pool.
            - shape (T_target, N): per-frame weights across N pools.
        window_size: Number of latent frames per matching window.
        hop: Stride between consecutive candidate pool windows.
    Returns:
        hybrid_latents: Weighted combination of the per-pool matched latents.
        match_indices: Tensor of shape (N, T_target) with the selected frame
            indices from each pool.
    """
    if not pool_files:
        raise ValueError("pool_files must contain at least one source pool")

    # Encode target once
    target_latents = encode_target(model, target_file, n_q)

    matched_latents = []
    match_indices = []

    # Encode each pool independently and match against the shared target
    for pool_file in pool_files:
        pool_latents = encode_pool(
            model, pool_file, n_q,
            pool_augmentations=pool_augmentations,
        )
        matched, indices = _match_windows(
            target_latents, pool_latents, window_size, hop,
            grain_size, stride,
        )
        matched_latents.append(matched)
        match_indices.append(indices)

    matched_stack = torch.stack(matched_latents, dim=0)
    match_index_stack = torch.stack(match_indices, dim=0)

    normalized_weights = _normalize_pool_weights(
        weights=weights,
        num_pools=len(pool_files),
        num_frames=matched_stack.shape[1],
        device=matched_stack.device,
        dtype=matched_stack.dtype,
    )

    hybrid_latents = (matched_stack * normalized_weights).sum(dim=0)

    return hybrid_latents, match_index_stack


def tokui_style_transfer_pool_morph(
    model,
    target_file: str,
    pool_file_a: str,
    pool_file_b: str,
    n_q: int = 8,
    morph: float | torch.Tensor = 0.0,
    window_size: int = 3,
    hop: int = 1,
    grain_size: int = 1,
    stride: int = 1,
    pool_augmentations: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Style transfer with morphing between two source pools.

    Each target frame is matched independently against pool_a and pool_b,
    then the resulting matched latents are blended by morph. This allows
    the timbral character of the output to transition continuously between
    two source corpora — either as a fixed mix or as a time-varying
    trajectory when morph is a tensor.

    Args:
        target_file:   Path to the target audio file whose temporal structure
                       is preserved in the output.
        pool_file_a:   Path to the first source pool. At morph=0.0 the output
                       is drawn entirely from this pool.
        pool_file_b:   Path to the second source pool. At morph=1.0 the output
                       is drawn entirely from this pool.
        n_q:           Number of RVQ quantizers used during encoding.
        morph:         Blending weight between the two pools in [0, 1].
                       - 0.0: output uses only pool_a matches.
                       - 1.0: output uses only pool_b matches.
                       - float: fixed mix across all frames.
                       - Tensor of shape (T_target, 1): per-frame morph weight,
                         enabling time-varying transitions such as slow crossfades,
                         LFO-driven oscillation, or onset-triggered jumps.
    """
    hybrid_latents, match_indices = tokui_style_transfer_multi_pool_morph(
        model=model,
        target_file=target_file,
        pool_files=[pool_file_a, pool_file_b],
        n_q=n_q,
        weights=_two_pool_weights(morph),
        window_size=window_size,
        hop=hop,
        grain_size=grain_size,
        stride=stride,
        pool_augmentations=pool_augmentations,
    )

    return hybrid_latents, match_indices[0], match_indices[1]