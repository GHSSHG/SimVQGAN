from __future__ import annotations

import threading
import queue
from typing import Iterable, Optional

import numpy as np
import jax
from flax.jax_utils import prefetch_to_device


class Prefetcher:
    """Threaded prefetcher for host-side numpy batches.

    - source_iter: yields batches (numpy arrays or pytrees)
    - prefetch_size: queue length
    - squeeze_channel_dim: if (B,1,L) -> (B,L)
    - dtype: output dtype (default float32)
    """

    def __init__(
        self,
        source_iter: Iterable[np.ndarray],
        prefetch_size: int = 8,
        squeeze_channel_dim: bool = True,
        dtype: np.dtype = np.float32,
    ):
        self._q: "queue.Queue[object]" = queue.Queue(maxsize=prefetch_size)
        self._SENTINEL = object()
        self._dtype = dtype
        self._stop = threading.Event()
        self._t = threading.Thread(
            target=self._worker, args=(source_iter, squeeze_channel_dim), daemon=True
        )
        self._t.start()

    def _worker(self, source_iter: Iterable[np.ndarray], squeeze_channel_dim: bool) -> None:
        try:
            for batch in source_iter:
                if self._stop.is_set():
                    break
                arr = np.asarray(batch)
                if squeeze_channel_dim and arr.ndim == 3 and arr.shape[1] == 1:
                    arr = arr.squeeze(1)
                if arr.dtype != self._dtype:
                    arr = np.asarray(arr, dtype=self._dtype)
                self._q.put(arr, block=True)
        except Exception as e:
            self._q.put(e)
        finally:
            self._q.put(self._SENTINEL)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is self._SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def join(self):
        if self._t.is_alive():
            self._t.join()

    def close(self) -> None:
        """Signal the worker to stop and drain the queue."""
        self._stop.set()
        # Drain pending batches so worker threads unstick even if the queue was full.
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(self._SENTINEL)
        except queue.Full:
            pass
        self.join()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()


def make_device_prefetcher(host_iter, device_prefetch_size=2, shard_for_multigpu=True, global_batch_size: Optional[int] = None):
    """Put batches onto device asynchronously; optionally shard across local devices."""
    ndev = jax.local_device_count()
    if shard_for_multigpu and ndev > 1 and global_batch_size is not None:
        if global_batch_size % ndev != 0:
            raise ValueError(f"Global batch size {global_batch_size} not divisible by number of devices {ndev}.")
    if shard_for_multigpu and ndev > 1:
        def _shard(batch):
            def _reshape(x):
                assert x.shape[0] % ndev == 0, f"global batch {x.shape[0]} not divisible by {ndev}"
                per = x.shape[0] // ndev
                return x.reshape((ndev, per) + x.shape[1:])
            return jax.tree_util.tree_map(_reshape, batch)
        sharded_iter = (_shard(b) for b in host_iter)
        return prefetch_to_device(sharded_iter, device_prefetch_size)
    else:
        devices = jax.local_devices()
        target = devices[0] if devices else None
        q: "queue.Queue[object]" = queue.Queue(maxsize=device_prefetch_size)
        sentinel = object()

        def _worker():
            try:
                for batch in host_iter:
                    if target is not None:
                        batch = jax.device_put(batch, target)
                    q.put(batch, block=True)
            except Exception as exc:  # pragma: no cover - surfacing worker errors
                q.put(exc)
            finally:
                q.put(sentinel)

        threading.Thread(target=_worker, daemon=True).start()

        def _iterator():
            while True:
                item = q.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

        return _iterator()
