"""
utils.py: Utility functions for Loading & Encoding audio using Encodec.

This toolkit provides access to three different levels of Encodec's internals:

1. The compression ladder:
   Use bandwidth_to_n_q / n_q_to_bandwidth to reconstruct audio at different codebook levels 
   via tokens_TN_to_audio_1T. This reveals what each additional RVQ level contributes 
   perceptually (e.g., coarse pitch at the bottom, fine timbre at the top).

2. The quantisation bottleneck:
   Compare audio_to_latents (continuous, pre-RVQ) against tokens_to_summary_latents 
   (post-quantisation, reconstructed from discrete codes). The difference reflects the 
   quantisation error.

3. Per-level codebook behaviour:
   token_level_to_latents lets you isolate what each n_q codebook encodes. 
   build_LOOKUP_via_layer_decode allows inspection of the codebook vectors at each level.

audio → encoder → RVQ → tokens → latents

"""
import torch
import torch.nn.functional as F
import numpy as np
from math import gcd
import soundfile as sf
from scipy.signal import resample_poly

# Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
def bandwidth_to_n_q(bw_kbps):
    """
    Translates from Encodec's bandwidth in kbps to the number of active codebook levels (n_q).
    Bandwidth in kbps (1.5, 3.0, 6.0, 12.0, 24.0) maps directly to number of active codebooks 
    (2, 4, 8, 16, 32). More codebooks = higher fidelity.
    
    For the 24 kHz model:
    - 1.5 kbps → 2 codebooks
    - 3.0 kbps → 4 codebooks
    - 6.0 kbps → 8 codebooks
    - 12.0 kbps → 16 codebooks
    - 24.0 kbps → 32 codebooks 

    For the 48 kHz model (only 3, 6, 12, 24 kbps supported), the number of codebooks is half 
    that of the 24 kHz model (e.g., 3 kbps → 2 codebooks).
    """
    return {1.5: 2, 3: 4, 6: 8, 12: 16, 24: 32}[bw_kbps]

def n_q_to_bandwidth(n_q):
    """
    Translates from number of active codebooks (n_q) to Encodec's bandwidth in kbps.
    The number of codebooks determines the resolution and quality of the encoded 
    audio representation.
    """
    return {2: 1.5, 4: 3, 8: 6, 16: 12, 32: 24}[n_q]
    

def load_wav_mono(path, target_sr=24000):
    """
    Reads an audio file and returns it as a mono waveform, ready for Encodec 
    which expects single-channel input.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, up=target_sr // g, down=sr // g).astype("float32")
    return torch.from_numpy(audio).unsqueeze(0)   # [1, T]

#=========================================
# for audio to tokens in different formats required ..

def tokens_TN_to_BQT(tokens_TN: torch.Tensor) -> torch.Tensor:
    # [T,N] -> [1,N,T]
    if tokens_TN.ndim != 2:
        raise ValueError(f"Expected [T,N], got {tuple(tokens_TN.shape)}")
    return tokens_TN.transpose(0, 1).unsqueeze(0).contiguous()

def tokens_BQT_to_TN(tokens_BQT: torch.Tensor) -> torch.Tensor:
    # [B,N,T] -> [T,N] using batch 0
    if tokens_BQT.ndim != 3:
        raise ValueError(f"Expected [B,N,T], got {tuple(tokens_BQT.shape)}")
    return tokens_BQT[0].transpose(0, 1).contiguous()

def tokens_BQT_to_QBT(tokens_BQT: torch.Tensor) -> torch.Tensor:
    # [B,N,T] -> [N,B,T]
    if tokens_BQT.ndim != 3:
        raise ValueError(f"Expected [B,N,T], got {tuple(tokens_BQT.shape)}")
    return tokens_BQT.permute(1, 0, 2).contiguous()

# -----
# and now the audio to token encoder

@torch.no_grad()
def encode_audio_to_tokens(audio, model, device, bandwidth, fmt: str = "TN", return_cpu: bool = False):
    """
    Runs the Encodec encoder on raw audio and returns a (T, n_q) token stack — 
    T frames, each represented by n_q codebook indices, one per RVQ level.

    fmt:
      - "TN":  [T_frames, n_q]      (recommended canonical)
      - "BQT": [B, n_q, T_frames]   (HF-friendly)
      - "QBT": [n_q, B, T_frames]   (for quantizer.decode)
    """
    import numpy as np
    import torch

    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    if not isinstance(audio, torch.Tensor):
        audio = torch.tensor(audio)
    audio = audio.float()

    # Force [B,C,T]
    if audio.ndim == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.ndim == 2:
        audio = audio.unsqueeze(1)
    elif audio.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected audio shape {tuple(audio.shape)}; need [T], [B,T], or [B,C,T].")

    audio = audio.to(device)
    model.eval()

    enc = model.encode(audio, bandwidth=bandwidth)
    codes = enc.audio_codes  # often [B,1,n_q,T]

    n_q=codes.shape[2]

    # Normalize to BQT = [B,n_q,T]
    if codes.ndim == 4:
        tokens_BQT = codes[:, 0].contiguous()
    elif codes.ndim == 3:
        tokens_BQT = codes.contiguous()
    else:
        raise ValueError(f"Unexpected audio_codes shape {tuple(codes.shape)}")

    if fmt.upper() == "BQT":
        out = tokens_BQT
    elif fmt.upper() == "QBT":
        out = tokens_BQT_to_QBT(tokens_BQT)
    elif fmt.upper() == "TN":
        out = tokens_BQT_to_TN(tokens_BQT)
    else:
        raise ValueError(f"fmt must be 'TN', 'BQT', or 'QBT' (got {fmt})")

    if return_cpu and isinstance(out, torch.Tensor):
        out = out.cpu()

    return out, enc



def load_ecdc(path: str):
    """
    Reads a pre-saved .ecdc file (Encodec's compressed format) and returns 
    the token stack, scales, and raw data, so you can skip re-encoding 
    if you've already processed audio.
    """
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict, got {type(obj)}")

    # Prefer your actual keys
    tokens = obj.get("audio_codes", None)
    scales = obj.get("audio_scales", None)

    if tokens is None:
        # fallback to other naming conventions
        tokens = obj.get("tokens", None) or obj.get("codes", None)
    if tokens is None:
        raise ValueError(f"Loaded {path} but couldn't find audio_codes/tokens/codes.")

    # tokens is (1, 1, N, T) in your file: (1,1,8,375)
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens)

    if tokens.ndim == 4:
        # (B, ?, N, T) -> take [0,0] -> (N,T) -> transpose -> (T,N)
        tokens = tokens[0, 0]
        tokens = tokens.transpose(0, 1)  # (T,N)
    elif tokens.ndim == 3:
        # common alt: (B,N,T) -> [0] -> (N,T) -> transpose
        tokens = tokens[0].transpose(0, 1)
    elif tokens.ndim == 2:
        # (T,N) or (N,T) heuristic: if first dim small, it's (N,T)
        if tokens.shape[0] <= 64 and tokens.shape[1] > tokens.shape[0]:
            tokens = tokens.transpose(0, 1)
    else:
        raise ValueError(f"Unexpected tokens shape {tuple(tokens.shape)}")

    tokens = tokens.long().contiguous()

    # scales in your file is [None]
    if isinstance(scales, list):
        scales = scales[0] if len(scales) > 0 else None
    if scales is not None and not isinstance(scales, torch.Tensor):
        scales = torch.tensor(scales).float()

    return tokens, scales, obj

#=============================================================
#=============================================================
# This is for building the token->latent lookup table

# ---- Helper: make sure audio is [B,C,T] ----
def ensure_BCT(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 1:
        return audio[None, None, :]
    if audio.ndim == 2:
        return audio[:, None, :]
    if audio.ndim == 3:
        return audio
    raise ValueError(f"Unexpected audio shape {tuple(audio.shape)}; expected [T], [B,T], or [B,C,T].")

# ---- Build lookup table by decoding each token index (RNeNcodec-style) ----
#  You only need to do this once!

@torch.no_grad()
def build_LOOKUP_via_layer_decode(model, n_q: int, K: int, device=None) -> torch.Tensor:
    """
    Builds a (n_q, K, 128) tensor where every entry [q, k] is the 128-d latent vector 
    you get when you decode token index k through codebook level q. 
    
    This is a one-time precomputation that lets you map any token to its latent 
    without running the decoder repeatedly: inspired by the RNEncodec approach.
    """
    if device is None:
        device = next(model.parameters()).device

    q = getattr(model, "quantizer", None)
    layers = getattr(q, "layers", None) or getattr(getattr(q, "vq", None), "layers", None)
    if layers is None:
        raise RuntimeError("encodec_model.quantizer.layers not found")

    idx_all = torch.arange(K, device=device, dtype=torch.long).unsqueeze(1)  # (K,1)

    E_list = []
    for qidx in range(n_q):
        z_kD1 = layers[qidx].decode(idx_all)  # expected (K,128,1)
        if z_kD1.ndim != 3 or z_kD1.shape[-1] != 1:
            raise RuntimeError(f"layers[{qidx}].decode(idx_all) returned {tuple(z_kD1.shape)}, expected (K,128,1)")
        E_list.append(z_kD1.squeeze(-1).contiguous())  # (K,128)

    return torch.stack(E_list, dim=0)  # (n_q,K,128)


# ---- Token stack (T,n_q) -> latent (T,128) via lookup+sum ----
# Can be used as a source "pool" of talents for style transfer purposes

@torch.no_grad()
def tokens_to_summary_latents(tokens_TN: torch.Tensor, LOOKUP_QKD: torch.Tensor) -> torch.Tensor:
    """
    Takes a (T, n_q) token stack and uses the lookup table to sum or average 
    across all n_q levels per frame, giving you a single (T, 128) latent sequence 
    that summarises the full RVQ representation at each timestep.
    """
    tokens_TN = tokens_TN.to(LOOKUP_QKD.device, dtype=torch.long)
    T, n_q = tokens_TN.shape
    assert LOOKUP_QKD.shape[0] >= n_q

    z_TD = torch.zeros(T, 128, device=LOOKUP_QKD.device, dtype=LOOKUP_QKD.dtype)
    for q in range(n_q):
        z_TD.add_(LOOKUP_QKD[q][tokens_TN[:, q]])
    return z_TD

# ---- Same as above, but only for one level of a token stack
@torch.no_grad()
def token_level_to_latents(tokens_TN: torch.Tensor, level_q: int, LOOKUP_QKD: torch.Tensor) -> torch.Tensor:
    """
    Returns latents for just one specific codebook level, giving you (T, 128) 
    representing only what that level encodes. Useful for per-level analysis as in the UMAP per-level plots.
    """
    tokens_TN = tokens_TN.to(LOOKUP_QKD.device, dtype=torch.long)
    return LOOKUP_QKD[level_q][tokens_TN[:, level_q]]  # (T,128)
    

@torch.no_grad()
def audio_to_latents(audio, model, device: str, bandwidth):
    """
    Encodes audio all the way to the continuous 128-d latent space (before quantisation), 
    returning (T, 128) floats. This is the pre-RVQ representation, which is what you want 
    for nearest-neighbour style transfer since it's continuous and smooth.
    """
    #codes_BQT, enc = audio_to_codes(audio, model, device=device, bandwidth=bandwidth)

    codes_BQT, enc =  encode_audio_to_tokens(audio, model, device=device, bandwidth=bandwidth, fmt="BQT")
    

    # HF quantizer.decode expects (n_q, B, T)
    codes_QBT = codes_BQT.permute(1, 0, 2).contiguous()

    z_BDT = model.quantizer.decode(codes_QBT)     # [B,128,T]
    z_BTD = z_BDT.permute(0, 2, 1).contiguous()   # [B,T,128]

    if z_BTD.shape[0] != 1:
        # If you ever pass batch>1, return the full batch tensor instead
        return z_BTD, codes_BQT.permute(0, 2, 1).contiguous(), enc

    z_T128 = z_BTD[0]                              # [T,128]
    codes_TQ = codes_BQT[0].transpose(0, 1).contiguous()  # [T,n_q]
    return z_T128, codes_TQ, enc

#====================================================
# Latents to audio
#----------------------------------------------------
@torch.no_grad()
def latents128_to_audio(model, z_T128, device):
    """
    Decodes a (T, 128) latent sequence back to audio by running it through 
    Encodec's decoder. This is the inverse of audio_to_latents.
    """
    model.eval()
    z_T128 = z_T128.to(device)

    # decoder expects [B, D, T_frames]
    z_BDT = z_T128.unsqueeze(0).transpose(1, 2).contiguous()  # [1,128,T]

    # Most HF builds expose decoder as model.decoder(...)
    audio_BCT = model.decoder(z_BDT)  # [1,C,T_samples]
    return audio_BCT[:, 0, :].contiguous()


#====================================================
# tokens_TN_to_audio_1T
#----------------------------------------------------

@torch.no_grad()
def tokens_TN_to_audio_1T(model, tokens_TN: torch.Tensor, device, audio_scales=None, last_frame_pad_length: int = 0):
    """
    Decodes a (T, n_q) token stack all the way back to audio, going through 
    the full quantizer decode + decoder pipeline. The last_frame_pad_length argument 
    handles any padding that was added during encoding to make frame counts work out.
    """
    model.eval()
    tokens_TN = tokens_TN.to(device, dtype=torch.long)

    # HF expects [B,1,N,T]
    codes_B1NT = tokens_TN.transpose(0, 1).unsqueeze(0).unsqueeze(0).contiguous()

    if audio_scales is None:
        audio_scales = [None]  # matches your .ecdc files

    audio_BCT = model.decode(
        audio_codes=codes_B1NT,
        audio_scales=audio_scales,
        last_frame_pad_length=last_frame_pad_length,
        padding_mask=None,
    )[0]  # [B,C,T]

    return audio_BCT[:, 0, :].detach().cpu().contiguous()
