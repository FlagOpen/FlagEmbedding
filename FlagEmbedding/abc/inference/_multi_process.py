import queue
import threading
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
            # Keep the sender alive until the parent consumes the failure and stops
            # the pool, including its Queue feeder and any outstanding results.
            continue
        results_queue.put([chunk_id, result])


def _watch_workers(pool, stopped, failures):
    while not stopped.wait(0.1):
        exited = [
            f"{process.pid} (exit code {process.exitcode})"
            for process in pool["processes"]
            if not process.is_alive()
        ]
        if not exited:
            continue
        failures.extend(exited)
        for process in pool["processes"]:
            if process.is_alive():
                process.terminate()
        for process in pool["processes"]:
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
        # Queue.get(timeout=...) cannot time out a partial frame. Once all child
        # writers are gone, close the parent's unused send handle too so recv sees
        # EOF. Queue.close() alone does not close it in a receive-only process.
        pool["output"]._writer.close()
        return


def collect_results(model, pool, num_chunks):
    """Collect ordered results, discarding a failed pool and any outstanding work."""
    stopped = threading.Event()
    failures = []
    watcher = threading.Thread(
        target=_watch_workers, args=(pool, stopped, failures), daemon=True,
        name="FlagEmbedding worker monitor",
    )
    watcher.start()
    try:
        try:
            results = []
            for _ in trange(num_chunks, desc="Chunks"):
                try:
                    chunk_id, result = pool["output"].get()
                except (EOFError, OSError) as error:
                    if failures:
                        raise RuntimeError(
                            "Inference worker exited before returning all results: " + ", ".join(failures)
                        ) from error
                    raise
                if isinstance(result, _WorkerError):
                    raise RuntimeError(
                        f"Inference worker failed while processing chunk {chunk_id}:\n{result.traceback}"
                    )
                results.append([chunk_id, result])
        finally:
            stopped.set()
            watcher.join()
        if failures:
            raise RuntimeError(
                "Inference worker exited before returning all results: " + ", ".join(failures)
            )
        return sorted(results, key=lambda item: item[0])
    except BaseException:
        # Pending input may exceed the pipe buffer after its consumers are terminated.
        pool["input"].cancel_join_thread()
        model.stop_multi_process_pool(pool)
        for attribute in ("pool", "query_pool"):
            if getattr(model, attribute, None) is pool:
                setattr(model, attribute, None)
        raise
