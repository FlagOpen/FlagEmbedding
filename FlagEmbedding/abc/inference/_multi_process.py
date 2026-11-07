import queue
import traceback
from typing import NamedTuple

from tqdm import trange


class _WorkerError(NamedTuple):
    traceback: str


def run_worker(compute, target_device, input_queue, results_queue):
    """Return an error as text: arbitrary exception objects may not be picklable."""
    while True:
        try:
            chunk_id, sentences, kwargs = input_queue.get()
        except queue.Empty:
            return
        try:
            result = compute(sentences, device=target_device, **kwargs)
        except Exception:
            results_queue.put([chunk_id, _WorkerError(traceback.format_exc())])
            return
        results_queue.put([chunk_id, result])


def collect_results(model, pool, num_chunks):
    """Collect ordered results, discarding a failed pool and any outstanding work."""
    try:
        results = []
        for _ in trange(num_chunks, desc="Chunks"):
            while True:
                try:
                    chunk_id, result = pool["output"].get(timeout=0.1)
                    break
                except queue.Empty:
                    exited = [
                        f"{process.pid} (exit code {process.exitcode})"
                        for process in pool["processes"]
                        if not process.is_alive()
                    ]
                    if exited:
                        # A worker can finish flushing its final message between the
                        # timed get and the liveness check.
                        try:
                            chunk_id, result = pool["output"].get_nowait()
                            break
                        except queue.Empty:
                            pass
                        raise RuntimeError(
                            "Inference worker exited before returning all results: " + ", ".join(exited)
                        )
            if isinstance(result, _WorkerError):
                raise RuntimeError(
                    f"Inference worker failed while processing chunk {chunk_id}:\n{result.traceback}"
                )
            results.append([chunk_id, result])
        return sorted(results, key=lambda item: item[0])
    except BaseException:
        # Pending input may exceed the pipe buffer after its consumers are terminated.
        pool["input"].cancel_join_thread()
        model.stop_multi_process_pool(pool)
        for attribute in ("pool", "query_pool"):
            if getattr(model, attribute, None) is pool:
                setattr(model, attribute, None)
        raise

