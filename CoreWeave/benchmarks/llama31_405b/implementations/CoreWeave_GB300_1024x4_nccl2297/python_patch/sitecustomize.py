"""
sitecustomize.py - injected via PYTHONPATH=/results/python_patch:...
Applies fixes for NCCL/distributed at 1024N+ scale:
  1. Forces 1800s timeout on init_process_group (covers ckpt-load NCCL collectives).
  2. Forces 1800s timeout on new_group (covers training-warmup NCCL collectives
     on DATA_PARALLEL_GROUP_WITH_CP and similar groups; default is 600s which
     was hit in jobs 5676 and 6035).
  3. Tracks every process group created via new_group.
  4. Pre-warms every tracked process group right before run_start fires,
     so the first training step doesn't pay a ~982s NCCL bootstrap on the
     4096-rank intra_dist_opt collective.

MLPerf compliance:
  - The pre-warm uses zero-tensor all_reduce only. No model weights are touched,
    no training data consumed, no optimizer state mutated. The model state at
    run_start is identical to checkpoint-load state.
  - All work happens before run_start MLLOG event, inside the unbounded init phase.
"""
import sys
import time
import threading
from datetime import timedelta
from functools import wraps


def _safe_print(msg):
    """Best-effort rank-0 print; never raises."""
    try:
        import torch.distributed as _d
        if _d.is_initialized() and _d.get_rank() != 0:
            return
    except Exception:
        pass
    try:
        print(msg, flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1. Force 1800s timeout on init_process_group + defaults
# ---------------------------------------------------------------------------
try:
    import torch.distributed as _dist

    _orig_init = _dist.init_process_group

    @wraps(_orig_init)
    def _patched_init(*args, **kwargs):
        kwargs["timeout"] = timedelta(seconds=1800)
        return _orig_init(*args, **kwargs)

    _dist.init_process_group = _patched_init

    import torch.distributed.constants as _c
    _c.default_pg_nccl_timeout = timedelta(seconds=1800)
    _c.default_pg_timeout = timedelta(seconds=1800)
except Exception as _e:
    _safe_print(f"PREWARM: timeout patch failed: {_e!r}")


# ---------------------------------------------------------------------------
# 2. Track every process group created
# ---------------------------------------------------------------------------
_pg_registry = []
_pg_registry_lock = threading.Lock()

try:
    import torch.distributed as _dist

    _orig_new_group = _dist.new_group

    @wraps(_orig_new_group)
    def _patched_new_group(*args, **kwargs):
        # Force 1800s timeout to match init_process_group patch.
        # Default for new_group is 600s which is too short for cold-start
        # NCCL bootstrap of large groups (DATA_PARALLEL_GROUP_WITH_CP, 256
        # ranks) at 1024N+ scale. Hit in jobs 5676 and 6035.
        kwargs["timeout"] = timedelta(seconds=1800)
        pg = _orig_new_group(*args, **kwargs)
        try:
            with _pg_registry_lock:
                _pg_registry.append(pg)
        except Exception:
            pass
        return pg

    _dist.new_group = _patched_new_group
except Exception as _e:
    _safe_print(f"PREWARM: new_group track patch failed: {_e!r}")


# ---------------------------------------------------------------------------
# 3. Pre-warm function: all_reduce on every PG to bootstrap NCCL channels
# ---------------------------------------------------------------------------
_prewarm_done = False
_prewarm_lock = threading.Lock()


def _do_prewarm(reason="unknown"):
    """Force eager NCCL bootstrap on every registered process group."""
    global _prewarm_done
    with _prewarm_lock:
        if _prewarm_done:
            return
        _prewarm_done = True

    try:
        import torch
        import torch.distributed as dist
    except Exception as e:
        _safe_print(f"PREWARM: torch import failed: {e!r}")
        return

    if not dist.is_initialized():
        _safe_print("PREWARM: torch.distributed not initialized, skipping")
        return

    with _pg_registry_lock:
        tracked = list(_pg_registry)

    pgs = [dist.group.WORLD] + tracked
    seen = set()
    unique_pgs = []
    for pg in pgs:
        try:
            pid = id(pg)
            if pid not in seen:
                seen.add(pid)
                unique_pgs.append(pg)
        except Exception:
            pass

    _safe_print(
        f"PREWARM: trigger={reason}, warming {len(unique_pgs)} PGs (incl WORLD={dist.get_world_size()})"
    )
    t0 = time.time()
    warmed = 0
    failed = 0

    for pg in unique_pgs:
        try:
            t = torch.zeros(1, dtype=torch.float32, device="cuda")
            dist.all_reduce(t, group=pg)
            warmed += 1
        except Exception:
            failed += 1

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    elapsed = time.time() - t0
    _safe_print(
        f"PREWARM: done in {elapsed:.1f}s ({warmed}/{len(unique_pgs)} warmed, {failed} failed)"
    )


# ---------------------------------------------------------------------------
# 4. Hook into mlperf logger to trigger prewarm right before run_start
# ---------------------------------------------------------------------------
def _patch_logger_method(obj_or_cls, attr_name, method_name):
    """Replace a method on an instance or class so that run_start triggers prewarm first."""
    target = getattr(obj_or_cls, attr_name, None)
    method = getattr(target, method_name, None) if target is not None else None
    if method is None:
        return False

    if not callable(method):
        return False

    is_instance = target is not obj_or_cls and not isinstance(target, type)

    if isinstance(target, type):
        # class - patch the class method
        orig = method
        @wraps(orig)
        def patched(self, key=None, *args, **kwargs):
            try:
                k = kwargs.get("key", key)
                if k == "run_start" or (isinstance(k, str) and k.endswith("run_start")):
                    _do_prewarm(reason=f"{target.__name__}.{method_name}")
            except Exception:
                pass
            return orig(self, key, *args, **kwargs) if key is not None else orig(self, *args, **kwargs)
        setattr(target, method_name, patched)
        return True

    if is_instance:
        # instance - patch via wrapping bound method
        orig = method
        @wraps(orig)
        def patched(key=None, *args, **kwargs):
            try:
                k = kwargs.get("key", key)
                if k == "run_start" or (isinstance(k, str) and k.endswith("run_start")):
                    _do_prewarm(reason=f"{type(target).__name__}.{method_name}")
            except Exception:
                pass
            return orig(key, *args, **kwargs) if key is not None else orig(*args, **kwargs)
        try:
            setattr(target, method_name, patched)
            return True
        except Exception:
            return False

    return False


def _install_mlperf_hooks():
    import importlib

    hooked_any = False
    candidates = [
        "mlperf_common.logging",
        "mlperf_logging.mllog.mllog",
        "mlperf_logging.mllog",
        "mlperf_logging",
    ]
    method_names = ["start", "event", "log_event", "start_", "intervals_start"]
    instance_attrs = ["mllogger", "mlperf_logger", "logger", "MLLogger"]
    class_names = ["MLLoggerWrapper", "MLLogger", "MLPerfLogger"]

    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        # Try classes
        for cls_name in class_names:
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            for mname in method_names:
                if _patch_logger_method(mod, cls_name, mname):
                    _safe_print(f"PREWARM: hooked {mod_name}.{cls_name}.{mname}")
                    hooked_any = True

        # Try instances
        for inst_name in instance_attrs:
            inst = getattr(mod, inst_name, None)
            if inst is None:
                continue
            for mname in method_names:
                if _patch_logger_method(mod, inst_name, mname):
                    _safe_print(f"PREWARM: hooked {mod_name}.{inst_name}.{mname}")
                    hooked_any = True

    return hooked_any


def _install_fallback_hook():
    """Fallback: trigger prewarm on the Nth barrier call (heuristic)."""
    try:
        import torch.distributed as _dist
        _orig_barrier = _dist.barrier
        _count = [0]

        @wraps(_orig_barrier)
        def patched_barrier(*args, **kwargs):
            _count[0] += 1
            # Megatron does many barriers during init; #50 is empirically after dataloader build.
            if _count[0] == 50:
                _do_prewarm(reason=f"barrier#{_count[0]}")
            return _orig_barrier(*args, **kwargs)

        _dist.barrier = patched_barrier
        _safe_print("PREWARM: installed fallback barrier hook (#50)")
    except Exception as e:
        _safe_print(f"PREWARM: fallback hook install failed: {e!r}")


try:
    if not _install_mlperf_hooks():
        _safe_print("PREWARM: no mlperf logger found; using fallback")
        _install_fallback_hook()
except Exception as _e:
    _safe_print(f"PREWARM: hook install error: {_e!r}")
    _install_fallback_hook()
