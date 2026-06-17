import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]


class TensorLikePrefixIndices:
    def __len__(self):
        return 4

    def __bool__(self):
        raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")


def load_stateweaver_metrics_module():
    path = REPO_ROOT / "python/sglang/srt/observability/stateweaver_metrics.py"
    spec = importlib.util.spec_from_file_location("stateweaver_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStateWeaverMetrics(unittest.TestCase):
    def test_cache_row_uses_request_prefix_and_cache_fields(self):
        module = load_stateweaver_metrics_module()
        req = SimpleNamespace(
            rid="req-123",
            origin_input_ids=list(range(10)),
            num_matched_prefix_tokens=6,
            stateweaver_radix_prefix_match_len=6,
            cached_tokens=6,
            cached_tokens_device=4,
            cached_tokens_host=2,
            cached_tokens_storage=0,
            extra_key="stateweaver-profile-a",
            routing_key="tenant/session/document",
            finished=lambda: True,
        )

        row = module.build_stateweaver_cache_row(req, {"device": 4, "host": 2})

        self.assertEqual(row["schema_version"], "stateweaver_cache_report_v1")
        self.assertEqual(row["request_id"], "req-123")
        self.assertEqual(row["input_tokens"], 10)
        self.assertEqual(row["matched_prefix_tokens"], 6)
        self.assertEqual(row["reused_kv_tokens"], 6)
        self.assertEqual(row["new_prefill_tokens"], 4)
        self.assertEqual(row["prefix_cache_hit_ratio"], 0.6)
        self.assertEqual(row["radix_prefix_match_len"], 6)
        self.assertEqual(row["hicache_l1_hit_tokens"], 4)
        self.assertEqual(row["hicache_l2_hit_tokens"], 2)
        self.assertEqual(row["cache_namespace"], "stateweaver-profile-a")
        self.assertEqual(row["routing_key"], "tenant/session/document")
        self.assertEqual(row["field_status"]["kv_fetch_time_ms"], "needs_hicache_hook")
        self.assertEqual(
            row["field_status"]["prefill_compute_time_ms"], "needs_runtime_hook"
        )

    def test_cache_row_reports_explicit_zero_radix_match_as_available(self):
        module = load_stateweaver_metrics_module()
        req = SimpleNamespace(
            rid="req-000",
            origin_input_ids=list(range(10)),
            num_matched_prefix_tokens=0,
            stateweaver_radix_prefix_match_len=0,
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            extra_key=None,
            routing_key=None,
            finished=lambda: True,
        )

        row = module.build_stateweaver_cache_row(req, None)

        self.assertEqual(row["matched_prefix_tokens"], 0)
        self.assertEqual(row["radix_prefix_match_len"], 0)
        self.assertEqual(row["field_status"]["matched_prefix_tokens"], "available_now")
        self.assertEqual(row["field_status"]["radix_prefix_match_len"], "available_now")

    def test_cache_row_derives_match_when_legacy_runtime_only_reports_reused_kv(self):
        module = load_stateweaver_metrics_module()
        req = SimpleNamespace(
            rid="req-456",
            origin_input_ids=list(range(10)),
            num_matched_prefix_tokens=0,
            prefix_indices=[],
            host_hit_length=0,
            cached_tokens=6,
            cached_tokens_device=6,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            extra_key=None,
            routing_key=None,
            finished=lambda: True,
        )

        row = module.build_stateweaver_cache_row(req, None)

        self.assertEqual(row["matched_prefix_tokens"], 6)
        self.assertEqual(row["radix_prefix_match_len"], 0)
        self.assertEqual(
            row["field_status"]["matched_prefix_tokens"],
            "derived_from_reused_kv_tokens",
        )
        self.assertEqual(
            row["field_status"]["radix_prefix_match_len"], "needs_runtime_hook"
        )

    def test_cache_row_does_not_use_finish_time_cache_state_as_raw_match(self):
        module = load_stateweaver_metrics_module()
        req = SimpleNamespace(
            rid="req-789",
            origin_input_ids=list(range(10)),
            num_matched_prefix_tokens=0,
            prefix_indices=[1, 2, 3, 4],
            host_hit_length=2,
            cached_tokens=6,
            cached_tokens_device=4,
            cached_tokens_host=2,
            cached_tokens_storage=0,
            extra_key=None,
            routing_key=None,
            finished=lambda: True,
        )

        row = module.build_stateweaver_cache_row(req, None)

        self.assertEqual(row["matched_prefix_tokens"], 6)
        self.assertEqual(row["radix_prefix_match_len"], 0)
        self.assertEqual(
            row["field_status"]["matched_prefix_tokens"],
            "derived_from_reused_kv_tokens",
        )
        self.assertEqual(
            row["field_status"]["radix_prefix_match_len"], "needs_runtime_hook"
        )

    def test_cache_row_handles_tensor_like_prefix_indices(self):
        module = load_stateweaver_metrics_module()
        req = SimpleNamespace(
            rid="req-tensor",
            origin_input_ids=list(range(10)),
            num_matched_prefix_tokens=0,
            prefix_indices=TensorLikePrefixIndices(),
            host_hit_length=2,
            cached_tokens=6,
            cached_tokens_device=4,
            cached_tokens_host=2,
            cached_tokens_storage=0,
            extra_key=None,
            routing_key=None,
            finished=lambda: True,
        )

        row = module.build_stateweaver_cache_row(req, None)

        self.assertEqual(row["radix_prefix_match_len"], 0)
        self.assertEqual(
            row["field_status"]["radix_prefix_match_len"], "needs_runtime_hook"
        )

    def test_server_args_declares_default_off_stateweaver_flags(self):
        text = (REPO_ROOT / "python/sglang/srt/server_args.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("enable_stateweaver_metrics: bool = False", text)
        self.assertIn("stateweaver_cache_report_path: Optional[str] = None", text)
        self.assertIn("stateweaver_expert_report_path: Optional[str] = None", text)
        self.assertIn('"--enable-stateweaver-metrics"', text)
        self.assertIn('"--stateweaver-cache-report-path"', text)
        self.assertIn('"--stateweaver-expert-report-path"', text)


if __name__ == "__main__":
    unittest.main()
