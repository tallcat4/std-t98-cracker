from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as st_load_file

from .sequence import generate_pn_sequence_196

SECRET_BURST_BYTES_AMBE_2450 = 28
SECRET_RESULT_NONE = 0
SECRET_RESULT_CURRENT_KEY = 1
SECRET_RESULT_GLOBAL_CACHE = 2
SECRET_RESULT_FULL_SEARCH = 3


BITS_PER_FRAME = 49
FRAMES_PER_BURST = 4
BURSTS_PER_BLOCK = 5
FRAMES_PER_BLOCK = FRAMES_PER_BURST * BURSTS_PER_BLOCK
INPUT_DIM = BITS_PER_FRAME * FRAMES_PER_BLOCK
HIDDEN_DIM = 128
NUM_CLASSES = 2
LABEL_PLAIN = 0
LABEL_ENCRYPTED = 1
MAX_KEY = 32767
HYBRID_BATCH_SIZE = 512

_THUMBDV_MAP = np.array(
    [
        0, 18, 36, 1, 19, 37, 2, 20, 38,
        3, 21, 39, 4, 22, 40, 5, 23, 41,
        6, 24, 42, 7, 25, 43, 8, 26, 44,
        9, 27, 45, 10, 28, 46, 11, 29, 47,
        12, 30, 48, 13, 31, 14, 32, 15, 33,
        16, 34, 17, 35,
    ],
    dtype=np.intp,
)


@dataclass(frozen=True)
class SecretResolution:
    resolved_key: int
    result_source: int
    cache_keys: tuple[int, ...]


class AMBE2Classifier(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(inputs)))


class _AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scores = self.attn(inputs).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return (inputs * weights.unsqueeze(-1)).sum(dim=1)


class AMBE2HybridClassifier(nn.Module):
    def __init__(
        self,
        n_frames: int = FRAMES_PER_BLOCK,
        bits_per_frame: int = BITS_PER_FRAME,
        cnn_channels: tuple[int, ...] = (64, 128, 256),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        classifier_hidden: int = 128,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self._n_frames = n_frames
        self._bits_per_frame = bits_per_frame

        in_ch = bits_per_frame
        cnn_blocks: list[nn.Module] = []
        for out_ch in cnn_channels:
            cnn_blocks.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2
        self.attn_pool = _AttentionPooling(lstm_out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 2:
            inputs = inputs.view(-1, self._n_frames, self._bits_per_frame)
        inputs = inputs.permute(0, 2, 1)
        inputs = self.cnn(inputs)
        inputs = inputs.permute(0, 2, 1)
        inputs, _ = self.lstm(inputs)
        inputs = self.attn_pool(inputs)
        return self.classifier(inputs)


def _load_checkpoint(model_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    metadata_path = model_path.with_suffix(".json")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if metadata.get("model_type") == "hybrid":
        model: nn.Module = AMBE2HybridClassifier(
            n_frames=metadata.get("n_frames", FRAMES_PER_BLOCK),
            bits_per_frame=metadata.get("bits_per_frame", BITS_PER_FRAME),
            cnn_channels=tuple(metadata.get("cnn_channels", [64, 128, 256])),
            lstm_hidden=metadata.get("lstm_hidden", 128),
            lstm_layers=metadata.get("lstm_layers", 2),
            classifier_hidden=metadata.get("classifier_hidden", 128),
            num_classes=metadata.get("num_classes", NUM_CLASSES),
            dropout=metadata.get("dropout", 0.3),
        )
    else:
        model = AMBE2Classifier(
            input_dim=metadata.get("input_dim", INPUT_DIM),
            hidden_dim=metadata.get("hidden_dim", HIDDEN_DIM),
            num_classes=metadata.get("num_classes", NUM_CLASSES),
        )

    weights = st_load_file(str(model_path), device=str(device))
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    return model, metadata


def _burst_payload_to_raw_bits(burst_payload: bytes) -> np.ndarray:
    frames = np.zeros((FRAMES_PER_BURST, BITS_PER_FRAME), dtype=np.uint8)
    for frame_index in range(FRAMES_PER_BURST):
        offset = frame_index * 7
        payload = burst_payload[offset : offset + 7]
        bits = []
        for byte_val in payload:
            for shift in range(7, -1, -1):
                bits.append((byte_val >> shift) & 1)
        frames[frame_index] = bits[:BITS_PER_FRAME]
    return frames


def _payload_to_blocks(payload: bytes, burst_count: int) -> list[np.ndarray]:
    bursts = [
        payload[index * SECRET_BURST_BYTES_AMBE_2450 : (index + 1) * SECRET_BURST_BYTES_AMBE_2450]
        for index in range(burst_count)
    ]
    raw_bursts = [_burst_payload_to_raw_bits(burst_payload) for burst_payload in bursts]

    blocks = []
    for start in range(0, len(raw_bursts) - BURSTS_PER_BLOCK + 1, BURSTS_PER_BLOCK):
        chunk = raw_bursts[start : start + BURSTS_PER_BLOCK]
        if len(chunk) == BURSTS_PER_BLOCK:
            blocks.append(np.vstack(chunk))
    return blocks[:2]


def _generate_all_pn_sequences(generate=generate_pn_sequence_196) -> np.ndarray:
    all_pn = np.zeros((MAX_KEY, 196), dtype=np.uint8)
    for key in range(1, MAX_KEY + 1):
        all_pn[key - 1] = generate(key)
    return all_pn


def _descramble_block_batch(raw_frames_block: np.ndarray, all_pn: np.ndarray) -> np.ndarray:
    num_keys = all_pn.shape[0]
    result = np.empty((num_keys, FRAMES_PER_BLOCK, BITS_PER_FRAME), dtype=np.uint8)

    for frame_index in range(raw_frames_block.shape[0]):
        pn_frame_index = frame_index % FRAMES_PER_BURST
        key_slice = all_pn[:, pn_frame_index * 49 : (pn_frame_index + 1) * 49]
        key_mapped = key_slice[:, _THUMBDV_MAP]
        result[:, frame_index, :] = raw_frames_block[frame_index][np.newaxis, :] ^ key_mapped

    return result


class SecretCracker:
    def __init__(
        self,
        ffnn_model_path: Path,
        hybrid_model_path: Path,
        *,
        device: str = "cpu",
        generate=generate_pn_sequence_196,
        batch_size: int = 4096,
        global_cache_limit: int = 16,
        cache_verify_limit: int = 2,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.global_cache_limit = global_cache_limit
        self.cache_verify_limit = cache_verify_limit
        self.ffnn_model, _ = _load_checkpoint(Path(ffnn_model_path), self.device)
        self.hybrid_model, _ = _load_checkpoint(Path(hybrid_model_path), self.device)
        self.all_pn = _generate_all_pn_sequences(generate)
        self.global_cache: list[int] = []

    def resolve_key(self, current_key: int, burst_payload: bytes, burst_count: int) -> SecretResolution:
        blocks_raw = _payload_to_blocks(burst_payload, burst_count)
        if not blocks_raw:
            return SecretResolution(0, SECRET_RESULT_NONE, tuple(self.global_cache))

        if current_key > 0:
            resolved_key = self._verify_cached_candidates(blocks_raw, [current_key])
            if resolved_key:
                self._remember_key(resolved_key)
                return SecretResolution(resolved_key, SECRET_RESULT_CURRENT_KEY, tuple(self.global_cache))

        cache_candidates = [key for key in self.global_cache if key != current_key][: self.cache_verify_limit]
        resolved_key = self._verify_cached_candidates(blocks_raw, cache_candidates)
        if resolved_key:
            self._remember_key(resolved_key)
            return SecretResolution(resolved_key, SECRET_RESULT_GLOBAL_CACHE, tuple(self.global_cache))

        resolved_key = self._full_search(blocks_raw)
        if resolved_key:
            self._remember_key(resolved_key)
            return SecretResolution(resolved_key, SECRET_RESULT_FULL_SEARCH, tuple(self.global_cache))

        return SecretResolution(0, SECRET_RESULT_NONE, tuple(self.global_cache))

    def _remember_key(self, key: int) -> None:
        if key <= 0:
            return
        self.global_cache = [cached for cached in self.global_cache if cached != key]
        self.global_cache.insert(0, key)
        del self.global_cache[self.global_cache_limit :]

    def _verify_cached_candidates(self, blocks_raw: list[np.ndarray], candidate_keys: list[int]) -> int:
        best_choice: tuple[int, float, int] | None = None
        for candidate_key in candidate_keys:
            if candidate_key <= 0 or candidate_key > MAX_KEY:
                continue
            plain_votes, best_margin = self._score_candidate_with_ffnn(blocks_raw, candidate_key)
            if plain_votes <= 0:
                continue
            choice = (plain_votes, best_margin, candidate_key)
            if best_choice is None or choice > best_choice:
                best_choice = choice

        return 0 if best_choice is None else best_choice[2]

    def _score_candidate_with_ffnn(self, blocks_raw: list[np.ndarray], candidate_key: int) -> tuple[int, float]:
        pn = self.all_pn[candidate_key - 1 : candidate_key]
        plain_votes = 0
        best_margin = float("-inf")

        with torch.no_grad():
            for block in blocks_raw:
                descrambled = _descramble_block_batch(block, pn).reshape(1, -1).astype(np.float32)
                logits = self.ffnn_model(torch.from_numpy(descrambled).to(self.device)).cpu()[0]
                if logits.argmax().item() == LABEL_PLAIN:
                    plain_votes += 1
                margin = float((logits[LABEL_PLAIN] - logits[LABEL_ENCRYPTED]).item())
                best_margin = max(best_margin, margin)

        return plain_votes, best_margin

    def _full_search(self, blocks_raw: list[np.ndarray]) -> int:
        with torch.no_grad():
            init_x = torch.from_numpy(blocks_raw[0].reshape(1, -1).astype(np.float32)).to(self.device)
            init_logits = self.hybrid_model(init_x)
            if init_logits.argmax().item() == LABEL_PLAIN:
                return 0

        desc_b0 = _descramble_block_batch(blocks_raw[0], self.all_pn).reshape(MAX_KEY, -1).astype(np.float32)
        desc_b1 = None
        if len(blocks_raw) > 1:
            desc_b1 = _descramble_block_batch(blocks_raw[1], self.all_pn).reshape(MAX_KEY, -1).astype(np.float32)

        preds_b0 = self._batched_argmax(self.ffnn_model, desc_b0, self.batch_size)
        if desc_b1 is None:
            candidate_mask = preds_b0 == LABEL_PLAIN
        else:
            preds_b1 = self._batched_argmax(self.ffnn_model, desc_b1, self.batch_size)
            candidate_mask = (preds_b0 == LABEL_PLAIN) | (preds_b1 == LABEL_PLAIN)

        candidates = np.where(candidate_mask)[0]
        if len(candidates) == 0:
            return 0

        scores_stage2, probs_stage2 = self._batched_plain_scores(desc_b0[candidates], HYBRID_BATCH_SIZE)
        survivors = np.where(scores_stage2 > 0)[0]
        if len(survivors) < 5:
            survivors = np.argsort(scores_stage2)[::-1][: min(5, len(scores_stage2))]
        if len(survivors) == 0:
            return 0

        stage2_candidates = candidates[survivors]
        stage2_scores = scores_stage2[survivors]
        stage2_probs = probs_stage2[survivors]

        if desc_b1 is None:
            best_index = int(np.argmax(stage2_scores))
            return int(stage2_candidates[best_index] + 1)

        scores_stage3, probs_stage3 = self._batched_plain_scores(desc_b1[stage2_candidates], HYBRID_BATCH_SIZE)
        ranking = sorted(
            range(len(stage2_candidates)),
            key=lambda index: (scores_stage3[index] > 0, stage2_scores[index], stage2_probs[index], probs_stage3[index]),
            reverse=True,
        )
        if not ranking:
            return 0
        return int(stage2_candidates[ranking[0]] + 1)

    def _batched_argmax(self, model: nn.Module, samples: np.ndarray, batch_size: int) -> np.ndarray:
        predictions = np.empty(samples.shape[0], dtype=np.int64)
        with torch.no_grad():
            for start in range(0, samples.shape[0], batch_size):
                end = min(start + batch_size, samples.shape[0])
                batch = torch.from_numpy(samples[start:end]).to(self.device)
                predictions[start:end] = model(batch).argmax(dim=1).cpu().numpy()
        return predictions

    def _batched_plain_scores(self, samples: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        logits_parts = []
        with torch.no_grad():
            for start in range(0, samples.shape[0], batch_size):
                end = min(start + batch_size, samples.shape[0])
                batch = torch.from_numpy(samples[start:end]).to(self.device)
                logits_parts.append(self.hybrid_model(batch).cpu())

        if not logits_parts:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

        logits = torch.cat(logits_parts, dim=0)
        probs = F.softmax(logits, dim=1)[:, LABEL_PLAIN].numpy()
        scores = (logits[:, LABEL_PLAIN] - logits[:, LABEL_ENCRYPTED]).numpy()
        return scores, probs
