from __future__ import annotations

import io
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import EncodecModel

from synthesis import _match_windows
from utils import load_wav_mono, n_q_to_bandwidth, audio_to_latents

ROOT = Path(__file__).resolve().parents[1]
WEBSITE_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBSITE_DIR / "static"
TEMPLATES_DIR = WEBSITE_DIR / "templates"

SAMPLE_RATE = 24_000


def _tokens_to_latents_via_quantizer(model: EncodecModel, tokens_TN: torch.Tensor) -> torch.Tensor:
    # tokens_TN: [T, N]
    codes_BQT = tokens_TN.transpose(0, 1).unsqueeze(0).contiguous()  # [1, N, T]
    codes_QBT = codes_BQT.permute(1, 0, 2).contiguous()  # [N, 1, T]
    z_BDT = model.quantizer.decode(codes_QBT)  # [1, 128, T]
    return z_BDT.permute(0, 2, 1).contiguous()[0]  # [T, 128]

_MODEL_LOCK = threading.Lock()
_MODEL: EncodecModel | None = None


def _get_encodec_model(device: str) -> EncodecModel:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=True)
            model.eval()
            model.to(device)
            _MODEL = model
        return _MODEL


def _latents_to_wav_bytes(model: EncodecModel, z_T128: torch.Tensor) -> bytes:
    z_BDT = z_T128.unsqueeze(0).transpose(1, 2).contiguous()
    audio_BCT = model.decoder(z_BDT)  # [1, C, T]
    audio = audio_BCT[:, 0, :].detach().cpu().numpy().squeeze()

    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@dataclass(frozen=True)
class DemoConfig:
    target_path: Path
    pool_path: Path
    n_q: int = 16
    window_size: int = 3
    hop: int = 1
    grain_size: int = 1
    stride: int = 1
    seconds: float = 4.0


class DemoEngine:
    def __init__(self, config: DemoConfig):
        self.config = config
        self._lock = threading.Lock()
        self._model: EncodecModel | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._prepared = False
        self._target_tokens: torch.Tensor | None = None
        self._matched_pool_tokens: torch.Tensor | None = None
        self._render_cache: dict[int, bytes] = {}

    def _load_model(self) -> EncodecModel:
        if self._model is None:
            self._model = _get_encodec_model(self._device)
        return self._model

    def _prepare(self) -> None:
        if self._prepared:
            return

        model = self._load_model()
        bandwidth = n_q_to_bandwidth(self.config.n_q)

        target_audio = load_wav_mono(str(self.config.target_path), target_sr=SAMPLE_RATE)
        pool_audio = load_wav_mono(str(self.config.pool_path), target_sr=SAMPLE_RATE)

        max_samples = int(self.config.seconds * SAMPLE_RATE)
        target_audio = target_audio[:, :max_samples].contiguous()
        pool_audio = pool_audio[:, :max_samples].contiguous()

        target_latents, target_tokens, _ = audio_to_latents(
            target_audio, model, self._device, bandwidth
        )
        pool_latents, pool_tokens, _ = audio_to_latents(
            pool_audio, model, self._device, bandwidth
        )

        matched_latents, match_indices = _match_windows(
            target_latents=target_latents,
            pool_latents=pool_latents,
            window_size=self.config.window_size,
            hop=self.config.hop,
            grain_size=self.config.grain_size,
            stride=self.config.stride,
        )
        _ = matched_latents  # matching is used only to pick pool frames

        match_indices = match_indices.clamp(0, pool_tokens.shape[0] - 1)
        matched_pool_tokens = pool_tokens[match_indices]

        if matched_pool_tokens.shape != target_tokens.shape:
            raise RuntimeError(
                "Token/frame misalignment in demo preparation: "
                f"target_tokens={tuple(target_tokens.shape)} "
                f"matched_pool_tokens={tuple(matched_pool_tokens.shape)}"
            )

        self._target_tokens = target_tokens.to(self._device)
        self._matched_pool_tokens = matched_pool_tokens.to(self._device)

        self._prepared = True

    def render(self, k_replace: int) -> bytes:
        with self._lock:
            self._prepare()

            n_q = int(self.config.n_q)
            k = int(max(0, min(n_q, k_replace)))
            if k in self._render_cache:
                return self._render_cache[k]

            assert self._target_tokens is not None
            assert self._matched_pool_tokens is not None

            model = self._load_model()

            hybrid_tokens = self._target_tokens.clone()
            if k > 0:
                hybrid_tokens[:, :k] = self._matched_pool_tokens[:, :k]

            z_T128 = _tokens_to_latents_via_quantizer(model, hybrid_tokens)
            payload = _latents_to_wav_bytes(model, z_T128)
            self._render_cache[k] = payload
            return payload


@dataclass(frozen=True)
class MorphConfig:
    target_path: Path
    pool_a_path: Path
    pool_b_path: Path
    n_q: int = 16
    window_size: int = 3
    hop: int = 1
    grain_size: int = 1
    stride: int = 1
    seconds: float = 4.0


class MorphEngine:
    """
    Cached two-pool morphing (Tokui-style): match target independently against
    pool A and pool B once, then blend matched latents by morph in [0, 1].
    """

    def __init__(self, config: MorphConfig):
        self.config = config
        self._lock = threading.Lock()
        self._model: EncodecModel | None = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._prepared = False
        self._matched_a: torch.Tensor | None = None
        self._matched_b: torch.Tensor | None = None
        self._render_cache: dict[int, bytes] = {}

    def _load_model(self) -> EncodecModel:
        if self._model is None:
            self._model = _get_encodec_model(self._device)
        return self._model

    def _prepare(self) -> None:
        if self._prepared:
            return

        model = self._load_model()
        bandwidth = n_q_to_bandwidth(self.config.n_q)

        max_samples = int(self.config.seconds * SAMPLE_RATE)
        target_audio = load_wav_mono(str(self.config.target_path), target_sr=SAMPLE_RATE)[:, :max_samples].contiguous()
        pool_a_audio = load_wav_mono(str(self.config.pool_a_path), target_sr=SAMPLE_RATE)[:, :max_samples].contiguous()
        pool_b_audio = load_wav_mono(str(self.config.pool_b_path), target_sr=SAMPLE_RATE)[:, :max_samples].contiguous()

        target_latents, _, _ = audio_to_latents(target_audio, model, self._device, bandwidth)
        pool_a_latents, _, _ = audio_to_latents(pool_a_audio, model, self._device, bandwidth)
        pool_b_latents, _, _ = audio_to_latents(pool_b_audio, model, self._device, bandwidth)

        matched_a, _ = _match_windows(
            target_latents=target_latents,
            pool_latents=pool_a_latents,
            window_size=self.config.window_size,
            hop=self.config.hop,
            grain_size=self.config.grain_size,
            stride=self.config.stride,
        )
        matched_b, _ = _match_windows(
            target_latents=target_latents,
            pool_latents=pool_b_latents,
            window_size=self.config.window_size,
            hop=self.config.hop,
            grain_size=self.config.grain_size,
            stride=self.config.stride,
        )

        self._matched_a = matched_a.to(self._device)
        self._matched_b = matched_b.to(self._device)
        self._prepared = True

    def render(self, morph: float) -> tuple[bytes, float]:
        with self._lock:
            self._prepare()
            m = float(max(0.0, min(1.0, morph)))
            key = int(round(m * 100.0))
            if key in self._render_cache:
                return self._render_cache[key], key / 100.0

            assert self._matched_a is not None
            assert self._matched_b is not None

            model = self._load_model()
            z = (1.0 - m) * self._matched_a + m * self._matched_b
            payload = _latents_to_wav_bytes(model, z)
            self._render_cache[key] = payload
            return payload, key / 100.0


templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="Latent Granular Resynthesis – Website")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


demo_engine = DemoEngine(
    DemoConfig(
        target_path=ROOT / "wav24k" / "percussion" / "amen_mono_24k.wav",
        pool_path=ROOT / "wav24k" / "percussion" / "tabla.wav",
    )
)

morph_engine = MorphEngine(
    MorphConfig(
        target_path=ROOT / "wav24k" / "percussion" / "amen_mono_24k.wav",
        pool_a_path=ROOT / "wav24k" / "percussion" / "djembe.wav",
        pool_b_path=ROOT / "wav24k" / "percussion" / "tabla.wav",
    )
)


@app.get("/", response_class=HTMLResponse)
def info(request: Request):
    return templates.TemplateResponse("info.html", {"request": request})


@app.get("/demo", response_class=HTMLResponse)
def demo(request: Request):
    return templates.TemplateResponse(
        "demo.html",
        {
            "request": request,
            "n_q": demo_engine.config.n_q,
            "seconds": demo_engine.config.seconds,
        },
    )

@app.get("/morph", response_class=HTMLResponse)
def morph(request: Request):
    return templates.TemplateResponse(
        "morph.html",
        {"request": request, "seconds": morph_engine.config.seconds},
    )


@app.get("/api/demo/render")
def render_demo(k: int = 0):
    try:
        payload = demo_engine.render(k_replace=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return Response(content=payload, media_type="audio/wav")


@app.get("/api/morph/render")
def render_morph(m: float = 0.0):
    try:
        payload, used = morph_engine.render(morph=m)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    # Return quantized morph used (0.00..1.00) so the client can display it if desired.
    return Response(content=payload, media_type="audio/wav", headers={"X-Morph-Used": f"{used:.2f}"})
