from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs


SCHEMA_VERSION = "stateweaver_cache_report_v1"
_WRITTEN_REQUEST_IDS: set[str] = set()


def maybe_write_stateweaver_cache_report(
    server_args: ServerArgs,
    req: Req,
    cached_tokens_details: Optional[dict[str, Any]],
) -> None:
    if (
        not server_args.enable_stateweaver_metrics
        or not server_args.stateweaver_cache_report_path
    ):
        return
    if hasattr(req, "finished") and not req.finished():
        return
    if req.rid in _WRITTEN_REQUEST_IDS:
        return

    path = Path(server_args.stateweaver_cache_report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = build_stateweaver_cache_row(req, cached_tokens_details)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    _WRITTEN_REQUEST_IDS.add(req.rid)


def build_stateweaver_cache_row(
    req: Req,
    cached_tokens_details: Optional[dict[str, Any]],
) -> dict[str, Any]:
    input_tokens = int(len(req.origin_input_ids))
    explicit_radix_prefix_match_len = getattr(
        req, "stateweaver_radix_prefix_match_len", None
    )
    raw_matched_prefix_tokens = int(
        explicit_radix_prefix_match_len
        if explicit_radix_prefix_match_len is not None
        else getattr(req, "num_matched_prefix_tokens", 0)
        or 0
    )
    reused_kv_tokens = int(getattr(req, "cached_tokens", 0) or 0)
    matched_prefix_tokens = raw_matched_prefix_tokens or reused_kv_tokens
    new_prefill_tokens = max(0, input_tokens - reused_kv_tokens)
    radix_status = (
        "available_now"
        if explicit_radix_prefix_match_len is not None
        or raw_matched_prefix_tokens
        or reused_kv_tokens == 0
        else "needs_runtime_hook"
    )
    prefix_status = (
        "available_now"
        if radix_status == "available_now"
        else "derived_from_reused_kv_tokens"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.time(),
        "request_id": req.rid,
        "input_tokens": input_tokens,
        "matched_prefix_tokens": matched_prefix_tokens,
        "reused_kv_tokens": reused_kv_tokens,
        "new_prefill_tokens": new_prefill_tokens,
        "prefix_cache_hit_ratio": (
            reused_kv_tokens / input_tokens if input_tokens > 0 else 0.0
        ),
        "radix_prefix_match_len": raw_matched_prefix_tokens,
        "hicache_l1_hit_tokens": int(getattr(req, "cached_tokens_device", 0) or 0),
        "hicache_l2_hit_tokens": int(getattr(req, "cached_tokens_host", 0) or 0)
        + int(getattr(req, "cached_tokens_storage", 0) or 0),
        "hicache_miss_tokens": new_prefill_tokens,
        "kv_fetch_time_ms": None,
        "prefill_compute_time_ms": None,
        "cache_namespace": req.extra_key or "",
        "routing_key": req.routing_key or "",
        "cached_tokens_details": cached_tokens_details,
        "field_status": {
            "request_id": "available_now",
            "input_tokens": "available_now",
            "matched_prefix_tokens": prefix_status,
            "reused_kv_tokens": "available_now",
            "new_prefill_tokens": "available_now",
            "prefix_cache_hit_ratio": "available_now",
            "radix_prefix_match_len": radix_status,
            "hicache_l1_hit_tokens": "available_now",
            "hicache_l2_hit_tokens": "available_now",
            "hicache_miss_tokens": "available_now",
            "kv_fetch_time_ms": "needs_hicache_hook",
            "prefill_compute_time_ms": "needs_runtime_hook",
            "cache_namespace": "available_now_or_config",
            "routing_key": "available_now_or_config",
        },
    }
