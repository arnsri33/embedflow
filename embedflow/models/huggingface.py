from __future__ import annotations

import gc
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from ..config import ModelConfig, hydrate_research_contract
from .base import EmbeddingModel


def _fingerprint(cfg: ModelConfig) -> str:
    semantic = {key: value for key, value in cfg.contract().items() if key not in {"local_path", "device"}}
    return hashlib.sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class HashEmbeddingModel(EmbeddingModel):
    """Deterministic tiny model used only by the self-contained demo/tests."""

    def __init__(self, model_id: str = "embedflow/demo-hash", dimension: int = 64):
        try:
            parsed_dimension = int(dimension)
            exact = float(dimension) == parsed_dimension
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("embedding dimension must be a positive integer") from exc
        if isinstance(dimension, bool) or not exact or parsed_dimension < 1:
            raise ValueError("embedding dimension must be a positive integer")
        self.model_id, self.dimension = str(model_id), parsed_dimension
        # Use the same semantic contract hashing as real adapters so demo
        # indexes, state files, and target caches agree on the model identity.
        self.fingerprint = ModelConfig(model_id, dimension=self.dimension).fingerprint

    def _encode(self, texts: list[str], role: str) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        rows = []
        for text in texts:
            # Demo vectors intentionally share a token-derived space across
            # query/document roles and model IDs, while fingerprints remain
            # distinct for cache safety. This makes the tiny demo useful for
            # retrieval without pretending to be a learned model.
            tokens = re.findall(r"[a-z0-9]+", str(text).lower()) or [str(text).lower()]
            pieces = []
            for token in tokens:
                seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")
                pieces.append(np.random.default_rng(seed).normal(size=self.dimension).astype("float32"))
            v = np.mean(pieces, axis=0).astype("float32")
            v /= max(float(np.linalg.norm(v)), 1e-12)
            rows.append(v)
        return np.asarray(rows, dtype="float32")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "query")

    def encode_documents(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        return self._encode(texts, "document")

    def close(self) -> None:
        return None


class HuggingFaceEmbeddingModel(EmbeddingModel):
    """HF/sentence-transformers adapter honoring the research contracts."""

    def __init__(self, cfg: ModelConfig, model_root: str | Path | None = None, device: str = "cpu"):
        self.cfg = hydrate_research_contract(cfg)
        self.model_id = self.cfg.model
        self.dimension = int(self.cfg.dimension or 0)
        self.fingerprint = _fingerprint(self.cfg)
        self.device = "cuda" if str(device).strip().lower() == "gpu" else str(device)
        model_path = Path(self.cfg.local_path or self.model_id)
        if model_root and not model_path.is_absolute():
            # Known research checkpoints are staged under models/<registry key>.
            registry = {
                "sentence-transformers/all-MiniLM-L6-v2": "minilm_l6",
                "Qwen/Qwen3-Embedding-0.6B": "qwen3_0_6b",
                "Qwen/Qwen3-Embedding-4B": "qwen3_4b",
                "Qwen/Qwen3-Embedding-8B": "qwen3_8b",
            }
            staged = Path(model_root) / registry.get(str(model_path), str(model_path))
            if staged.exists():
                model_path = staged
        self.model_path = model_path
        self._encoder = None
        self._closed = False
        self._kind = "sentence" if self.model_id.startswith("sentence-transformers/") else "transformers"
        self._load()
        self.cfg.dimension = self.dimension
        self.fingerprint = _fingerprint(self.cfg)

    def _load(self) -> None:
        # Reuse the project's frozen implementation where available. It has
        # the exact Qwen prompt, padding and last-token pooling behavior.
        try:
            known_contract = self.model_id in {
                "sentence-transformers/all-MiniLM-L6-v2", "Qwen/Qwen3-Embedding-0.6B",
                "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B"
            }
            if not known_contract or not self.model_path.exists():
                raise ImportError("use generic Hugging Face loader")
            from src.embed import make_encoder  # type: ignore
            contract = self.cfg.contract()
            contract["model_id"] = self.model_id
            contract["revision"] = self.cfg.revision
            contract["dimension"] = self.dimension
            self._encoder = make_encoder(contract, self.model_path, smoke=False, seed=0, device=self.device)
            if not self.dimension and hasattr(self._encoder, "dimension"):
                self.dimension = int(self._encoder.dimension)
            if not self.dimension and hasattr(self._encoder, "model") and hasattr(self._encoder.model, "get_sentence_embedding_dimension"):
                self.dimension = int(self._encoder.model.get_sentence_embedding_dimension())
            if not self.dimension and hasattr(self._encoder, "net"):
                self.dimension = int(getattr(getattr(self._encoder.net, "config", None), "hidden_size", 0) or 0)
            if not self.dimension:
                raise ValueError(f"could not infer embedding dimension for {self.model_id}; set source/target.dimension")
            return
        except (ImportError, ModuleNotFoundError):
            pass
        if self._kind == "sentence":
            from sentence_transformers import SentenceTransformer
            local_only = self.model_path.exists()
            self._encoder = SentenceTransformer(str(self.model_path), device=self.device,
                                                revision=self.cfg.revision, local_files_only=local_only)
            self._encoder.max_seq_length = self.cfg.max_length
            if not self.dimension:
                self.dimension = int(self._encoder.get_sentence_embedding_dimension())
        else:
            import torch
            from transformers import AutoModel, AutoTokenizer
            local_only = self.model_path.exists()
            self._tok = AutoTokenizer.from_pretrained(str(self.model_path), revision=self.cfg.revision,
                                                      local_files_only=local_only)
            self._tok.padding_side = self.cfg.padding_side
            self._tok.truncation_side = self.cfg.truncation_side
            self._tok.pad_token = self._tok.pad_token or self._tok.eos_token
            dtype = torch.bfloat16 if self.device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
            self._encoder = AutoModel.from_pretrained(str(self.model_path), revision=self.cfg.revision,
                                                      local_files_only=local_only, torch_dtype=dtype).eval().to(self.device)
            if not self.dimension:
                self.dimension = int(getattr(self._encoder.config, "hidden_size", 0) or getattr(self._encoder.config, "d_model", 0))
        if not self.dimension:
            raise ValueError(f"could not infer embedding dimension for {self.model_id}; set source/target.dimension")

    def _format(self, texts: list[str], role: str) -> list[str]:
        instruction = self.cfg.query_instruction if role == "query" else self.cfg.document_instruction
        return [instruction.replace("{text}", str(t)) if instruction else str(t) for t in texts]

    def _generic_encode(self, texts: list[str], role: str) -> np.ndarray:
        if self._kind == "sentence":
            vals = self._encoder.encode(self._format(texts, role), convert_to_numpy=True,
                                        normalize_embeddings=self.cfg.normalization.lower() == "l2",
                                        show_progress_bar=False, batch_size=min(32, max(1, len(texts))))
            return np.asarray(vals, dtype="float32")
        import torch
        formatted = self._format(texts, role)
        enc = self._tok(formatted, padding=True, truncation=True, max_length=self.cfg.max_length, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._encoder(**enc).last_hidden_state
            if self.cfg.pooling in {"last_token", "last_non_padding_token"}:
                # For left padding the final sequence position is the last
                # non-padding token; for right padding use the attention-mask
                # length.  This preserves the Qwen contract in both modes.
                if self.cfg.padding_side == "left":
                    positions = torch.full((out.shape[0],), out.shape[1] - 1, dtype=torch.long, device=out.device)
                else:
                    positions = enc["attention_mask"].sum(dim=1) - 1
                vals = out[torch.arange(out.shape[0], device=out.device), positions]
            else:
                mask = enc["attention_mask"].unsqueeze(-1)
                vals = (out * mask).sum(1) / mask.sum(1).clamp_min(1)
            vals = vals.float()
            if self.cfg.normalization.lower() == "l2":
                vals = torch.nn.functional.normalize(vals, dim=1)
        return vals.detach().cpu().numpy().astype("float32")

    def _encode(self, texts: list[str], role: str, batch_size: int | None = None) -> np.ndarray:
        if self._closed:
            raise RuntimeError("embedding model is closed")
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        if self._encoder is not None and hasattr(self._encoder, "encode") and not hasattr(self, "_tok"):
            # ``src.embed`` applies the frozen query/document instruction
            # internally; formatting here would prepend it twice.
            values, _ = self._encoder.encode(texts, role)
            return np.asarray(values, dtype="float32")
        return self._generic_encode(texts, role)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "query")

    def encode_documents(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if batch_size is None or len(texts) <= batch_size:
            return self._encode(texts, "document", batch_size)
        chunks = [self._encode(texts[i:i + batch_size], "document", batch_size) for i in range(0, len(texts), batch_size)]
        return np.concatenate(chunks, axis=0)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if hasattr(self, "_encoder"):
                del self._encoder
            if hasattr(self, "_tok"):
                del self._tok
        finally:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except Exception:
                pass


def load_embedding_model(cfg: ModelConfig, model_root: str | Path | None = None, device: str = "cpu",
                         demo: bool = False) -> EmbeddingModel:
    if not isinstance(cfg, ModelConfig):
        raise TypeError("cfg must be a ModelConfig")
    device = str(device or cfg.device or "cpu")
    if demo or cfg.model.startswith("embedflow/demo"):
        return HashEmbeddingModel(cfg.model or "embedflow/demo-hash", cfg.dimension or 64)
    if not cfg.model:
        raise ValueError("model identifier is required")
    return HuggingFaceEmbeddingModel(cfg, model_root=model_root, device=device)
