"""
IDLM Block-N pure-draft decoding (no introspective verify).

Unlike IDLMBlockN which uses block_size = 2N-1 with N-1 verify positions
holding speculative tokens from the previous round, this scheme uses
block_size = N. Every position in the input is MASK; the model fills
all of them in one forward and we directly commit the first `k` samples
(optionally truncated by a confidence threshold). KV cache provides the
full conditioning context — no specs carried across rounds.

Trade-off vs IDLMBlockN:
  + Forward size N instead of 2N-1 (43% smaller for N=4)
  + Per-forward commit ceiling is N (vs N for verify-accept-all, N for forced-cold)
  + No state machine between V / R / C — every forward is the same
  - No spec-style conditioning (specs from prev round) — quality depends purely
    on bidirectional MASK filling

Config keys (YAML via --dllm-algorithm-config):
  block_size:                  int,   MUST equal gen_block_size
  gen_block_size:              int,   tokens per step. Default 4.
  temperature, top_k, top_p:   sampling params
  confidence_accept_threshold: float, 0.0 = commit all N. Default 0.0.
                               When > 0, commits the longest prefix where each
                               sampled token has softmax-max-prob >= threshold,
                               with a floor of 1 (always commit at least 1).
  stats_file, rounds_file:     analysis dump paths. Defaults to
                               /tmp/idlm_stats.jsonl, /tmp/idlm_rounds.jsonl.
                               Same schema as IDLMBlockN so analysis scripts
                               work unchanged.
"""

import logging
import os
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class IDLMBlockNDraftOnly(DllmAlgorithm):
    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.gen_block_size: int = config.algorithm_config.get("gen_block_size", 4)
        assert self.block_size == self.gen_block_size, (
            f"IDLMBlockNDraftOnly requires block_size == gen_block_size, "
            f"got block_size={self.block_size}, gen_block_size={self.gen_block_size}"
        )
        self.temperature: float = config.algorithm_config.get("temperature", 1.0)
        self.top_k: int = config.algorithm_config.get("top_k", 50)
        self.top_p: float = config.algorithm_config.get("top_p", 0.95)
        self.confidence_accept_threshold: float = config.algorithm_config.get(
            "confidence_accept_threshold", 0.0
        )

        if "stats_file" in config.algorithm_config:
            self._stats_file = config.algorithm_config["stats_file"] or None
        else:
            self._stats_file = os.environ.get(
                "IDLM_STATS_FILE", "/tmp/idlm_stats.jsonl"
            )
        if "rounds_file" in config.algorithm_config:
            self._rounds_file = config.algorithm_config["rounds_file"] or None
        else:
            self._rounds_file = os.environ.get(
                "IDLM_ROUNDS_FILE", "/tmp/idlm_rounds.jsonl"
            )
        if self._stats_file:
            logger.info(f"[IDLMBlockNDraftOnly] per-request stats → {self._stats_file}")
        if self._rounds_file:
            logger.info(f"[IDLMBlockNDraftOnly] per-round signal trace → {self._rounds_file}")

        self._stats = {"total_forwards": 0, "total_tokens": 0}
        self._req_stats: Dict[int, Dict[str, int]] = {}
        self._req_rounds: Dict[int, list] = {}
        self._prev_last_clean_conf: Dict[int, float] = {}

        # Outputs the scheduler/output_processor consumes
        self._dllm_write_override: Dict[int, List[int]] = {}
        self._kv_trim_info: Dict[int, dict] = {}
        self._advance_override: Dict[int, int] = {}

        logger.info(
            f"[IDLMBlockNDraftOnly] block_size={self.block_size}, "
            f"conf_accept_thr={self.confidence_accept_threshold}, "
            f"temperature={self.temperature}"
        )

    def cleanup_request(self, req_pool_idx: int, rid: str = None):
        self._prev_last_clean_conf.pop(req_pool_idx, None)
        s = self._req_stats.pop(req_pool_idx, None)
        rounds = self._req_rounds.pop(req_pool_idx, None)

        if s is not None and s["n_forwards"] > 0:
            avg_commit = s["n_committed"] / s["n_forwards"]
            logger.info(
                f"[IDLMBlockNDraftOnly][req] rpx={req_pool_idx} rid={rid} "
                f"forwards={s['n_forwards']} committed={s['n_committed']} "
                f"tok/fwd={avg_commit:.2f}/{self.block_size}"
            )
            if self._stats_file:
                try:
                    import json as _json
                    line = {
                        "rid": rid,
                        "rpx": req_pool_idx,
                        # Reuse schema fields so math500 script displays correctly
                        "avg_accept_len": avg_commit,
                        "verify_num_specs": self.block_size,
                        "verify_rounds": s["n_forwards"],
                        "committed": s["n_committed"],
                        "forwards": s["n_forwards"],
                        "tok_per_fwd": avg_commit,
                    }
                    with open(self._stats_file, "a") as _fp:
                        _fp.write(_json.dumps(line) + "\n")
                except Exception as _e:
                    logger.warning(f"stats_file write failed: {_e}")

        if self._rounds_file and rounds:
            try:
                import json as _json
                with open(self._rounds_file, "a") as _fp:
                    for round_idx, r in enumerate(rounds):
                        line = {
                            "rid": rid,
                            "rpx": req_pool_idx,
                            "round": round_idx,
                            "pre_conf": r["pre_conf"],
                            "accept_len": r["accept_len"],
                            "case": r["case"],
                            "committed": r["committed"],
                            "verify_num_specs": self.block_size,
                        }
                        _fp.write(_json.dumps(line) + "\n")
            except Exception as _e:
                logger.warning(f"rounds_file write failed: {_e}")

    def _bump(self, rpx: int, committed: int):
        s = self._req_stats.get(rpx)
        if s is None:
            s = {"n_forwards": 0, "n_committed": 0}
            self._req_stats[rpx] = s
        s["n_forwards"] += 1
        s["n_committed"] += committed

    def _record(self, rpx: int, pre_conf, accept_len: int, committed: int):
        if self._rounds_file is None:
            return
        rounds = self._req_rounds.get(rpx)
        if rounds is None:
            rounds = []
            self._req_rounds[rpx] = rounds
        rounds.append(
            {
                "pre_conf": pre_conf,
                "accept_len": accept_len,
                "case": "D",
                "committed": committed,
            }
        )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        overlap_fn=None,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        blk = self.block_size  # = N

        _el = forward_batch.extend_seq_lens
        if _el is not None:
            extend_lens_cpu = (
                _el.tolist() if isinstance(_el, torch.Tensor) else list(_el)
            )
        else:
            extend_lens_cpu = [blk] * batch_size
        base_offsets = [0] * batch_size
        for bid in range(1, batch_size):
            base_offsets[bid] = base_offsets[bid - 1] + extend_lens_cpu[bid - 1]
        is_prefill = [extend_lens_cpu[bid] != blk for bid in range(batch_size)]

        # Cached CPU values from prepare_for_dllm_decode
        _cached_rpx = getattr(forward_batch, "dllm_rpx_cpu", None)
        if _cached_rpx is not None and len(_cached_rpx) == batch_size:
            req_pool_indices_cpu = _cached_rpx
            seq_lens_cpu = forward_batch.dllm_seq_lens_cpu
        else:
            _combined = torch.cat([
                forward_batch.req_pool_indices[:batch_size],
                forward_batch.seq_lens[:batch_size].to(forward_batch.req_pool_indices.dtype),
            ])
            _combined_cpu = _combined.tolist()
            req_pool_indices_cpu = _combined_cpu[:batch_size]
            seq_lens_cpu = [int(x) for x in _combined_cpu[batch_size:]]

        # Pure prefill path
        if any(el == blk for el in extend_lens_cpu):
            has_any_decode = True
        else:
            has_any_decode = (forward_batch.input_ids == self.mask_id).any().item()
        if not has_any_decode:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            self._stats["total_forwards"] += 1
            return out.logits_output, [], out.can_run_graph

        # Clear per-step outputs
        self._dllm_write_override.clear()
        self._kv_trim_info.clear()
        self._advance_override.clear()

        # Decode: input is already all-MASK from prep. Forward directly.
        forward_batch.dllm_force_causal = True
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        forward_batch.dllm_force_causal = False

        if overlap_fn is not None:
            overlap_fn()

        logits_output = out.logits_output
        full_logits = logits_output.full_logits

        decode_bids = [bid for bid in range(batch_size) if not is_prefill[bid]]

        # Sample at all block positions for all decode bids in one batched call.
        # Each decode bid contributes `blk` consecutive positions.
        next_token_ids_list: List = [None] * batch_size
        if decode_bids:
            all_idx = []
            for bid in decode_bids:
                base = base_offsets[bid]
                for j in range(blk):
                    all_idx.append(base + j)
            idx_t = torch.tensor(all_idx, dtype=torch.long, device=device)
            block_logits = full_logits[idx_t]  # [nd*blk, vocab]

            # Sample + confidence at each position
            if self.temperature <= 0:
                samples = block_logits.argmax(dim=-1)
                # max-softmax-prob serves as confidence proxy
                conf = F.softmax(block_logits, dim=-1).gather(
                    1, samples.unsqueeze(1)
                ).squeeze(1)
            else:
                if self.temperature != 1.0:
                    scaled = block_logits / self.temperature
                else:
                    scaled = block_logits
                probs = F.softmax(scaled, dim=-1)
                # Optional top_k/top_p; for simplicity, mask scaled logits.
                if self.top_k > 0:
                    topk_vals, _ = scaled.topk(self.top_k, dim=-1)
                    scaled = scaled.masked_fill(
                        scaled < topk_vals[:, -1:], float("-inf")
                    )
                if self.top_p < 1.0:
                    sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
                    sm = sorted_logits.softmax(dim=-1)
                    cum = sm.cumsum(dim=-1)
                    cutoff = (cum - sm) >= self.top_p
                    sorted_logits = sorted_logits.masked_fill(
                        cutoff, float("-inf")
                    )
                    scaled = torch.full_like(scaled, float("-inf")).scatter(
                        1, sorted_idx, sorted_logits
                    )
                sample_probs = F.softmax(scaled, dim=-1)
                samples = torch.multinomial(sample_probs, num_samples=1).squeeze(1)
                # Confidence uses the ORIGINAL (un-truncated) softmax max-prob, since
                # that's the calibrated quantity we compared against threshold in
                # IDLMBlockN's confidence-accept path. Comparable across modes.
                conf = probs.gather(1, samples.unsqueeze(1)).squeeze(1)

            samples_cpu = samples.tolist()
            conf_cpu = conf.tolist()
            tau = self.confidence_accept_threshold

            req_to_token = model_runner.req_to_token_pool.req_to_token
            all_trim_rpx: List[int] = []
            all_trim_pos: List[int] = []
            trim_offsets_per_bid: Dict[int, Tuple[int, int]] = {}

            for k_idx, bid in enumerate(decode_bids):
                rpx = req_pool_indices_cpu[bid]
                start = k_idx * blk
                req_samples = samples_cpu[start:start + blk]
                req_confs = conf_cpu[start:start + blk]

                if tau > 0:
                    accept_k = 0
                    for j in range(blk):
                        if req_confs[j] >= tau:
                            accept_k += 1
                        else:
                            break
                    if accept_k == 0:
                        accept_k = 1  # ensure forward progress
                else:
                    accept_k = blk

                output_tokens = req_samples[:accept_k]
                dllm_tokens = list(output_tokens) + [self.mask_id] * (blk - accept_k)

                next_token_ids_list[bid] = output_tokens
                self._dllm_write_override[rpx] = dllm_tokens
                self._advance_override[rpx] = accept_k

                trim_count = blk - accept_k
                if trim_count > 0:
                    sl = seq_lens_cpu[bid]
                    off_start = len(all_trim_rpx)
                    for t in range(trim_count):
                        all_trim_rpx.append(rpx)
                        all_trim_pos.append(sl - 1 - t)
                    trim_offsets_per_bid[bid] = (off_start, trim_count)

                pre_conf = self._prev_last_clean_conf.get(rpx)
                self._bump(rpx, committed=accept_k)
                if self._rounds_file is not None:
                    self._record(
                        rpx,
                        pre_conf=pre_conf,
                        accept_len=accept_k,
                        committed=accept_k,
                    )
                    # Pre-conf for NEXT round = the last accepted token's conf
                    self._prev_last_clean_conf[rpx] = float(req_confs[accept_k - 1])

            # Batched KV-index lookup
            if all_trim_rpx:
                all_kv_indices = req_to_token[all_trim_rpx, all_trim_pos]
            else:
                all_kv_indices = None
            for bid in decode_bids:
                rpx = req_pool_indices_cpu[bid]
                if bid in trim_offsets_per_bid:
                    off_start, tc = trim_offsets_per_bid[bid]
                    self._kv_trim_info[rpx] = {
                        "kv_indices_gpu": all_kv_indices[off_start:off_start + tc],
                        "trim_count": tc,
                    }
                else:
                    self._kv_trim_info[rpx] = {
                        "kv_indices_gpu": None,
                        "trim_count": 0,
                    }

        # Inline prefill bids: no commit, empty next_token list
        for bid in range(batch_size):
            if next_token_ids_list[bid] is None:
                next_token_ids_list[bid] = []

        # Per-step debug log (single-request only, mirrors IDLMBlockN's [STEP] line)
        if batch_size == 1 and not is_prefill[0] and decode_bids:
            bid0 = decode_bids[0]
            rpx0 = req_pool_indices_cpu[bid0]
            start = 0  # k_idx=0 for single request
            req_samples = samples_cpu[start:start + blk]
            req_confs = conf_cpu[start:start + blk]
            adv = self._advance_override.get(rpx0, "?")
            tc = self._kv_trim_info.get(rpx0, {}).get("trim_count", "?")
            out_toks = next_token_ids_list[bid0]
            conf_str = "[" + ",".join(f"{c:.2f}" for c in req_confs) + "]"
            logger.info(
                f"[STEP] D → out={out_toks} adv={adv} trim={tc} "
                f"samples={req_samples} conf={conf_str}"
            )

        self._stats["total_forwards"] += 1
        self._stats["total_tokens"] += sum(
            len(t) for t in next_token_ids_list if t is not None
        )
        return logits_output, next_token_ids_list, out.can_run_graph


Algorithm = IDLMBlockNDraftOnly
