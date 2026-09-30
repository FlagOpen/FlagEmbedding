"""Regression tests for the batch-size probing policy (issue #1607).

The single-device inference paths probe for a workable batch size by running a
trial forward pass and shrinking the batch size on failure. Previously the
handler caught *every* RuntimeError and retried with ``batch_size * 3 // 4``
with no lower bound: a non-OOM error was swallowed, batch size 1 became 0 on
the first retry, and persistent errors looped at zero forever.

The fixed policy (helpers in ``FlagEmbedding/inference/utils.py``):
  * only genuine OOM errors are retried; other RuntimeErrors propagate
    immediately with their original traceback;
  * retries stop (re-raising the OOM error) when the batch size would drop
    below 1, so it never reaches 0;
  * non-positive caller-supplied batch sizes are rejected with ValueError.

These tests are torch-free: the helpers module is loaded directly from its
file path (importing the ``FlagEmbedding`` package itself needs torch), and
the inference source files are checked with the ``ast`` module, mirroring the
repro approach from the issue.
"""

import ast
import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
UTILS_PATH = os.path.join(REPO_ROOT, "FlagEmbedding", "inference", "utils.py")

# The 9 files carrying a batch-size probing loop, and the function(s) in
# each that probe.
PROBE_SITES = {
    "FlagEmbedding/inference/embedder/encoder_only/base.py": ["encode_single_device"],
    "FlagEmbedding/inference/embedder/encoder_only/m3.py": ["encode_single_device"],
    "FlagEmbedding/inference/embedder/decoder_only/base.py": ["encode_single_device"],
    "FlagEmbedding/inference/embedder/decoder_only/icl.py": [
        "encode_queries_single_device",
        "encode_single_device",
    ],
    "FlagEmbedding/inference/embedder/decoder_only/pseudo_moe.py": ["encode_single_device"],
    "FlagEmbedding/inference/reranker/encoder_only/base.py": ["compute_score_single_gpu"],
    "FlagEmbedding/inference/reranker/decoder_only/base.py": ["compute_score_single_gpu"],
    "FlagEmbedding/inference/reranker/decoder_only/layerwise.py": ["compute_score_single_gpu"],
    "FlagEmbedding/inference/reranker/decoder_only/lightweight.py": ["compute_score_single_gpu"],
}


def _load_utils():
    spec = importlib.util.spec_from_file_location("flag_inference_utils", UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = _load_utils()


def _probe_handlers(tree):
    """Yield ``except RuntimeError`` handlers that adjust ``batch_size``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        caught = ast.unparse(node.type)
        if "RuntimeError" not in caught:
            continue
        touches_batch = any(
            isinstance(sub, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "batch_size" for t in sub.targets)
            for sub in ast.walk(node)
        )
        calls_shrink = any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "shrink_batch_size_on_oom"
            for sub in ast.walk(node)
        )
        if touches_batch or calls_shrink:
            yield node


class HelperBehaviorTests(unittest.TestCase):
    def test_is_oom_error_detects_cuda_oom(self):
        self.assertTrue(utils.is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 1.5 GiB")))

    def test_is_oom_error_detects_cpu_oom(self):
        self.assertTrue(
            utils.is_oom_error(RuntimeError("DefaultCPUAllocator: not enough memory: you tried to allocate 4GB"))
        )

    def test_is_oom_error_rejects_non_oom(self):
        self.assertFalse(utils.is_oom_error(RuntimeError("shape mismatch in matmul")))
        self.assertFalse(utils.is_oom_error(RuntimeError("CUDA error: device-side assert triggered")))

    def test_shrink_sequence_never_drops_below_one(self):
        sizes = [256]
        err = RuntimeError("CUDA out of memory")
        while True:
            nxt = sizes[-1] * 3 // 4
            if nxt < 1:
                break
            sizes.append(utils.shrink_batch_size_on_oom(sizes[-1], err))
        self.assertEqual(sizes[-1], 1)
        self.assertTrue(all(s >= 1 for s in sizes))
        self.assertEqual(sizes, [256, 192, 144, 108, 81, 60, 45, 33, 24, 18, 13, 9, 6, 4, 3, 2, 1])

    def test_shrink_reraises_original_error_at_single_sample(self):
        err = RuntimeError("CUDA out of memory")
        with self.assertRaises(RuntimeError) as ctx:
            utils.shrink_batch_size_on_oom(1, err)
        self.assertIs(ctx.exception, err)

    def test_validate_batch_size_rejects_non_positive(self):
        for bad in (0, -1, -256):
            with self.assertRaises(ValueError, msg=f"batch_size={bad}"):
                utils.validate_batch_size(bad)

    def test_validate_batch_size_accepts_positive_and_none(self):
        utils.validate_batch_size(1)
        utils.validate_batch_size(256)
        utils.validate_batch_size(None)


class FixedLoopSemanticsTests(unittest.TestCase):
    """Drive the fixed handler policy (same code shape as the patched
    inference loops) with stubbed trials, no torch needed."""

    @staticmethod
    def run_probe(batch_size, trial):
        sizes = []
        while True:
            sizes.append(batch_size)
            try:
                trial(batch_size)
                return batch_size, sizes
            except RuntimeError as e:
                if not utils.is_oom_error(e):
                    raise
                batch_size = utils.shrink_batch_size_on_oom(batch_size, e)

    def test_non_oom_error_propagates_immediately_with_original_traceback(self):
        boom = RuntimeError("some non-OOM failure")
        calls = []

        def trial(bs):
            calls.append(bs)
            raise boom

        with self.assertRaises(RuntimeError) as ctx:
            self.run_probe(32, trial)
        self.assertIs(ctx.exception, boom)
        self.assertEqual(calls, [32], "non-OOM error must not be retried")

    def test_oom_retries_shrink_until_success(self):
        def trial(bs):
            if bs > 4:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.0 GiB")

        final, sizes = self.run_probe(32, trial)
        self.assertEqual(final, 4)
        self.assertEqual(sizes, [32, 24, 18, 13, 9, 6, 4])
        self.assertTrue(all(s >= 1 for s in sizes))

    def test_persistent_oom_raises_at_batch_size_one(self):
        oom = RuntimeError("CUDA out of memory")
        sizes = []

        def trial(bs):
            sizes.append(bs)
            raise oom

        with self.assertRaises(RuntimeError) as ctx:
            self.run_probe(2, trial)
        self.assertIs(ctx.exception, oom)
        self.assertEqual(sizes, [2, 1], "must stop after single-sample attempt, never 0")


class ProbeSiteStructureTests(unittest.TestCase):
    """AST checks over every probing loop: they must route non-OOM errors to
    ``raise`` and shrink via ``shrink_batch_size_on_oom`` only. These fail on
    the pre-fix tree (handlers blindly assigned ``batch_size * 3 // 4``)."""

    def test_all_probe_handlers_use_fixed_policy(self):
        for relpath in PROBE_SITES:
            path = os.path.join(REPO_ROOT, relpath)
            with open(path) as fh:
                tree = ast.parse(fh.read())
            handlers = list(_probe_handlers(tree))
            self.assertTrue(handlers, f"no probe handler found in {relpath}")
            for handler in handlers:
                dump = ast.dump(handler)
                has_is_oom = any(
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "is_oom_error"
                    for sub in ast.walk(handler)
                )
                has_bare_raise = any(
                    isinstance(sub, ast.Raise) and sub.exc is None for sub in ast.walk(handler)
                )
                def _assigns_batch_directly(sub):
                    if not isinstance(sub, ast.Assign):
                        return False
                    if not any(isinstance(t, ast.Name) and t.id == "batch_size" for t in sub.targets):
                        return False
                    # the fixed policy assigns via shrink_batch_size_on_oom(...)
                    v = sub.value
                    return not (
                        isinstance(v, ast.Call)
                        and isinstance(v.func, ast.Name)
                        and v.func.id == "shrink_batch_size_on_oom"
                    )

                direct_assign = any(_assigns_batch_directly(sub) for sub in ast.walk(handler))
                self.assertTrue(has_is_oom, f"{relpath}:{handler.lineno}: handler must call is_oom_error()")
                self.assertTrue(has_bare_raise, f"{relpath}:{handler.lineno}: handler must re-raise non-OOM errors")
                self.assertFalse(
                    direct_assign,
                    f"{relpath}:{handler.lineno}: handler must not assign batch_size directly; "
                    "use shrink_batch_size_on_oom()",
                )

    def test_all_probe_functions_validate_batch_size(self):
        for relpath, fnames in PROBE_SITES.items():
            path = os.path.join(REPO_ROOT, relpath)
            with open(path) as fh:
                tree = ast.parse(fh.read())
            for fname in fnames:
                fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fname]
                self.assertEqual(len(fns), 1, f"{relpath}: expected one {fname}")
                calls = [
                    sub
                    for sub in ast.walk(fns[0])
                    if isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id == "validate_batch_size"
                ]
                self.assertTrue(calls, f"{relpath}: {fname} must call validate_batch_size()")

    def test_no_unbounded_probe_loop_remains(self):
        for relpath in PROBE_SITES:
            path = os.path.join(REPO_ROOT, relpath)
            with open(path) as fh:
                tree = ast.parse(fh.read())
            for handler in _probe_handlers(tree):
                src = ast.unparse(handler)
                self.assertNotIn(
                    "batch_size * 3 // 4",
                    src,
                    f"{relpath}:{handler.lineno}: raw shrink expression must be gone",
                )


if __name__ == "__main__":
    unittest.main()
