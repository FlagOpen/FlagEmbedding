import subprocess
import sys
from textwrap import dedent


def test_reranker_cleanup_during_collection_and_shutdown():
    result = subprocess.run(
        [sys.executable, "-c", dedent("""
            import gc
            import os
            from FlagEmbedding.abc.inference.AbsReranker import AbsReranker

            class Probe(AbsReranker):
                def compute_score_single_gpu(self, *args, **kwargs):
                    pass

                def stop_self_pool(self, write=os.write):
                    write(1, self.marker)

            def probe(marker):
                instance = object.__new__(Probe)
                instance.marker = marker
                instance.cycle = instance
                return instance

            probe(b'collected\\n')
            gc.collect()
            retained_until_shutdown = probe(b'shutdown\\n')
        """)],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.splitlines() == ["collected"]
    assert "Exception ignored" not in result.stderr
