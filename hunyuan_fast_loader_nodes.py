"""
HunyuanImage-3.0 Instruct Fast Loader Nodes for ComfyUI

Two alternative loader nodes that replace the standard safetensors I/O with
faster backends, giving significant speedups on systems with high storage
bandwidth or unified memory (e.g. DGX Spark, Grace Hopper):

  - HunyuanInstructLoaderInstantTensor  -- uses InstantTensor (CUDA-only,
    multi-file batched direct I/O, up to ~30x faster on NVMe)
  - HunyuanInstructLoaderFastSafetensors -- uses FastSafetensors (CPU+CUDA,
    automatic unified-memory copier on DGX Spark / Grace Hopper)

Both nodes have the exact same interface as HunyuanInstructLoader and share
its model cache, so they are fully interchangeable.

How it works
------------
transformers' ``from_pretrained`` (with ``device_map``) loads safetensors
via ``safetensors.safe_open`` -- NOT ``safetensors.torch.load_file``.  The
call chain is:

    from_pretrained -> _load_pretrained_model
        -> safe_open(shard, framework="pt", device=...)
        -> file_pointer.get_slice(name) -> param[...]

We monkey-patch ``safetensors.safe_open`` (and the imported reference in
``transformers.modeling_utils``) with a drop-in replacement backed by
the fast library.  A ``_FakeSafeOpen`` object serves pre-loaded tensors
through the same ``.get_tensor()`` / ``.get_slice()`` / ``.keys()`` /
``.metadata()`` API that transformers expects.
"""

import contextlib
import glob
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# -- availability checks (no hard failures at import time) ----------------

INSTANTTENSOR_AVAILABLE = False
try:
    from instanttensor import safe_open as _it_safe_open  # noqa: F401
    INSTANTTENSOR_AVAILABLE = True
except ImportError:
    pass

FASTSAFETENSORS_AVAILABLE = False
try:
    from fastsafetensors import fastsafe_open as _fst_open  # noqa: F401
    FASTSAFETENSORS_AVAILABLE = True
except ImportError:
    pass

# -- imports from the existing instruct nodes -----------------------------

try:
    from .hunyuan_instruct_nodes import (
        HunyuanInstructLoader,
        resolve_model_path,
        _instruct_cache,
    )
    _PARENT_AVAILABLE = True
except ImportError:
    try:
        from hunyuan_instruct_nodes import (
            HunyuanInstructLoader,
            resolve_model_path,
            _instruct_cache,
        )
        _PARENT_AVAILABLE = True
    except ImportError:
        _PARENT_AVAILABLE = False


# =========================================================================
# Torch dtype <-> safetensors dtype string mapping
# =========================================================================

_TORCH_TO_ST_DTYPE = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}


# =========================================================================
# Drop-in replacement classes for safetensors.safe_open
# =========================================================================

class _FakeSlice:
    """Mimics the ``PySafeSlice`` returned by ``safe_open.get_slice()``.

    Supports ``get_dtype()``, ``get_shape()``, and ``__getitem__``
    (e.g. ``slice[...]`` to materialise the full tensor).
    """

    __slots__ = ("_tensor",)

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def get_dtype(self) -> str:
        return _TORCH_TO_ST_DTYPE.get(self._tensor.dtype, "F32")

    def get_shape(self) -> List[int]:
        return list(self._tensor.shape)

    def __getitem__(self, key):
        return self._tensor[key]


class _FakeSafeOpen:
    """Drop-in replacement for ``safetensors.safe_open``.

    Works both as a context manager (``with safe_open(...) as f:``) and as
    a plain object (``f = safe_open(...)``), matching the two usage patterns
    in ``transformers.modeling_utils``.
    """

    def __init__(self, tensors: Dict[str, torch.Tensor],
                 file_metadata: Optional[Dict[str, str]] = None):
        self._tensors = tensors
        self._metadata = file_metadata or {}

    # -- context-manager protocol -----------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    # -- safetensors API surface used by transformers ---------------------
    def keys(self) -> List[str]:
        return list(self._tensors.keys())

    def metadata(self) -> Optional[Dict[str, str]]:
        return dict(self._metadata) if self._metadata else None

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]

    def get_slice(self, name: str) -> _FakeSlice:
        return _FakeSlice(self._tensors[name])


# =========================================================================
# Helpers
# =========================================================================

def _get_safetensors_files(model_path: str) -> List[str]:
    """Return sorted list of *.safetensors files in *model_path*."""
    return sorted(glob.glob(os.path.join(model_path, "*.safetensors")))


def _normalize_device_str(device) -> str:
    """Normalise a device argument to a string like 'cpu' or 'cuda:0'."""
    if device is None:
        return "cpu"
    s = str(device)
    if s.isdigit():
        return f"cuda:{s}"
    return s


def _read_file_metadata(filename: str, original_safe_open) -> Dict[str, str]:
    """Read only the JSON header metadata from a safetensors file (fast)."""
    try:
        with original_safe_open(filename, framework="pt") as f:
            return f.metadata() or {}
    except Exception:
        return {}


def _read_file_keys(filename: str, original_safe_open) -> List[str]:
    """Read tensor key names from a safetensors file header (fast)."""
    with original_safe_open(filename, framework="pt") as f:
        return list(f.keys())


# -- monkey-patching safe_open --------------------------------------------

def _patch_safe_open(fake_fn):
    """Replace ``safetensors.safe_open`` and the local reference imported in
    ``transformers.modeling_utils`` with *fake_fn*.

    Also patches ``safetensors.torch.load_file`` and the transformers alias
    ``safe_load_file`` so that the secondary ``load_sharded_checkpoint`` path
    is also covered.

    Returns (patches, original_safe_open).
    """
    import safetensors

    original = safetensors.safe_open
    patches: List[Tuple[Any, str, Any]] = []

    # 1. safetensors.safe_open (module-level binding)
    patches.append((safetensors, "safe_open", original))
    safetensors.safe_open = fake_fn

    # 2. transformers.modeling_utils.safe_open (copied at import time via
    #    ``from safetensors import safe_open``)
    try:
        import transformers.modeling_utils as tmu
        if hasattr(tmu, "safe_open"):
            patches.append((tmu, "safe_open", tmu.safe_open))
            tmu.safe_open = fake_fn
    except (ImportError, AttributeError):
        pass

    # 3. Also patch safetensors.torch.load_file for the secondary loading
    #    path (load_sharded_checkpoint) which uses ``safe_load_file``.
    try:
        import safetensors.torch as sft
        orig_load_file = sft.load_file
        patches.append((sft, "load_file", orig_load_file))

        def _replacement_load_file(filename, device="cpu"):
            with fake_fn(filename, framework="pt", device=device) as f:
                return {k: f.get_tensor(k) for k in f.keys()}

        sft.load_file = _replacement_load_file

        # The local alias in transformers.modeling_utils
        try:
            import transformers.modeling_utils as tmu
            if hasattr(tmu, "safe_load_file"):
                patches.append((tmu, "safe_load_file", tmu.safe_load_file))
                tmu.safe_load_file = _replacement_load_file
        except (ImportError, AttributeError):
            pass
    except (ImportError, AttributeError):
        pass

    return patches, original


def _restore_patches(patches):
    """Undo all patches applied by ``_patch_safe_open``."""
    for module, attr_name, original in patches:
        setattr(module, attr_name, original)


# =========================================================================
# InstantTensor context manager
# =========================================================================

@contextlib.contextmanager
def _instant_tensor_context(model_path: str):
    """Patch ``safetensors.safe_open`` with an InstantTensor-backed loader.

    All safetensors shards are bulk-loaded to CUDA in a single
    ``instanttensor.safe_open`` call (multi-file batched direct I/O).
    Pre-loading is **lazy**: it only happens when ``safe_open`` is actually
    called, so cached models skip the I/O entirely.
    """
    from instanttensor import safe_open as it_safe_open

    st_files = _get_safetensors_files(model_path)
    if not st_files:
        yield
        return

    st_abs_paths = {os.path.abspath(f) for f in st_files}

    # -- lazy pre-load state ----------------------------------------------
    all_tensors: Dict[str, torch.Tensor] = {}
    file_metadata: Dict[str, Dict[str, str]] = {}
    file_keys: Dict[str, List[str]] = {}
    preloaded = False

    # Capture the *original* safe_open before we install our patch.
    import safetensors as _sf
    original_safe_open = _sf.safe_open

    def _ensure_preloaded():
        nonlocal preloaded
        if preloaded:
            return
        preloaded = True

        # Read headers (metadata + key lists) -- fast, no tensor data
        for f in st_files:
            absp = os.path.abspath(f)
            file_metadata[absp] = _read_file_metadata(f, original_safe_open)
            file_keys[absp] = _read_file_keys(f, original_safe_open)

        total_keys = sum(len(v) for v in file_keys.values())
        logger.info(
            "[InstantTensor] Pre-loading %d file(s) (%d tensors) to CUDA ...",
            len(st_files), total_keys,
        )
        t0 = time.time()
        with it_safe_open(st_files, framework="pt", device=0) as f:
            for name, tensor in f.tensors():
                all_tensors[name] = tensor.clone()
        elapsed = time.time() - t0
        total_gb = sum(t.nbytes for t in all_tensors.values()) / (1024 ** 3)
        logger.info(
            "[InstantTensor] Loaded %d tensors (%.1f GB) in %.1f s  (%.1f GB/s)",
            len(all_tensors), total_gb, elapsed,
            total_gb / elapsed if elapsed > 0 else float("inf"),
        )

    # -- replacement safe_open --------------------------------------------
    def fake_safe_open(filename, framework="pt", device="cpu", **kw):
        absp = os.path.abspath(filename)
        if absp not in st_abs_paths:
            # Not one of the model shards -> fall through to original
            return original_safe_open(filename, framework=framework,
                                      device=device, **kw)
        _ensure_preloaded()
        keys = file_keys.get(absp, [])
        meta = file_metadata.get(absp, {})
        tensors: Dict[str, torch.Tensor] = {}
        # NOTE: We deliberately ignore the requested ``device`` argument and
        # return tensors at their preload location (CUDA). InstantTensor's
        # purpose is fast direct-to-CUDA loading -- copying back to CPU would
        # double memory usage (severe on UMA devices like DGX Spark / GH200,
        # where CPU and CUDA share the same physical RAM).  Downstream code
        # that needs tensors on a specific device should use ``tensor.device``
        # or call ``.to(device)`` itself when actually needed.
        for k in keys:
            if k in all_tensors:
                tensors[k] = all_tensors[k]
        return _FakeSafeOpen(tensors, meta)

    patches, _ = _patch_safe_open(fake_safe_open)
    try:
        yield
    finally:
        _restore_patches(patches)
        all_tensors.clear()
        file_metadata.clear()
        file_keys.clear()


# =========================================================================
# FastSafetensors context manager
# =========================================================================

@contextlib.contextmanager
def _fastsafetensors_context(model_path: str):
    """Patch ``safetensors.safe_open`` with a FastSafetensors-backed loader.

    Each shard is loaded individually via ``fastsafe_open``.  On DGX Spark /
    Grace Hopper the library automatically activates its unified-memory copier.
    """
    from fastsafetensors import fastsafe_open

    st_files = _get_safetensors_files(model_path)
    if not st_files:
        yield
        return

    st_abs_paths = {os.path.abspath(f) for f in st_files}

    # Capture original before patching
    import safetensors as _sf
    original_safe_open = _sf.safe_open

    def fake_safe_open(filename, framework="pt", device="cpu", **kw):
        absp = os.path.abspath(filename)
        if absp not in st_abs_paths:
            return original_safe_open(filename, framework=framework,
                                      device=device, **kw)

        dev = _normalize_device_str(device)
        meta = _read_file_metadata(filename, original_safe_open)

        t0 = time.time()
        tensors: Dict[str, torch.Tensor] = {}
        with fastsafe_open(filenames=[filename], device=dev, nogds=True) as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key).clone().detach()
        elapsed = time.time() - t0
        total_mb = sum(t.nbytes for t in tensors.values()) / (1024 ** 2)
        logger.info(
            "[FastSafetensors] %s -> %d tensors (%.0f MB) in %.1f s",
            os.path.basename(filename), len(tensors), total_mb, elapsed,
        )
        return _FakeSafeOpen(tensors, meta)

    patches, _ = _patch_safe_open(fake_safe_open)
    try:
        yield
    finally:
        _restore_patches(patches)


# =========================================================================
# Node: HunyuanInstructLoaderInstantTensor
# =========================================================================

if _PARENT_AVAILABLE:

    class HunyuanInstructLoaderInstantTensor(HunyuanInstructLoader):
        """Load HunyuanImage-3.0-Instruct model with **InstantTensor** backend.

        Identical interface to the standard Instruct Loader but replaces the
        safetensors I/O layer with InstantTensor's multi-file direct-I/O
        loader.  All tensors are pre-loaded to CUDA in one batched pass before
        ``from_pretrained`` is called, giving up to ~30x faster model loading
        on NVMe / tmpfs storage.

        Requires: ``pip install instanttensor``
        """

        @classmethod
        def INPUT_TYPES(cls):
            return HunyuanInstructLoader.INPUT_TYPES()

        def load_model(
            self,
            model_name: str,
            force_reload: bool = False,
            attention_impl: str = "sdpa",
            moe_impl: str = "eager",
            vram_reserve_gb: float = 30.0,
            blocks_to_swap: int = 0,
        ) -> Tuple[Any]:
            if not INSTANTTENSOR_AVAILABLE:
                raise RuntimeError(
                    "InstantTensor is not installed. "
                    "Install with:  pip install instanttensor"
                )
            model_path = resolve_model_path(model_name)
            with _instant_tensor_context(model_path):
                return super().load_model(
                    model_name, force_reload, attention_impl,
                    moe_impl, vram_reserve_gb, blocks_to_swap,
                )


    # =====================================================================
    # Node: HunyuanInstructLoaderFastSafetensors
    # =====================================================================

    class HunyuanInstructLoaderFastSafetensors(HunyuanInstructLoader):
        """Load HunyuanImage-3.0-Instruct model with **FastSafetensors** backend.

        Identical interface to the standard Instruct Loader but replaces the
        safetensors I/O layer with FastSafetensors.  On DGX Spark / Grace
        Hopper the library automatically activates its unified-memory copier
        for optimal throughput.

        Requires: ``pip install fastsafetensors``
        """

        @classmethod
        def INPUT_TYPES(cls):
            return HunyuanInstructLoader.INPUT_TYPES()

        def load_model(
            self,
            model_name: str,
            force_reload: bool = False,
            attention_impl: str = "sdpa",
            moe_impl: str = "eager",
            vram_reserve_gb: float = 30.0,
            blocks_to_swap: int = 0,
        ) -> Tuple[Any]:
            if not FASTSAFETENSORS_AVAILABLE:
                raise RuntimeError(
                    "fastsafetensors is not installed. "
                    "Install with:  pip install fastsafetensors"
                )
            model_path = resolve_model_path(model_name)
            with _fastsafetensors_context(model_path):
                return super().load_model(
                    model_name, force_reload, attention_impl,
                    moe_impl, vram_reserve_gb, blocks_to_swap,
                )


# =========================================================================
# ComfyUI registration
# =========================================================================

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if _PARENT_AVAILABLE:
    NODE_CLASS_MAPPINGS["HunyuanInstructLoaderInstantTensor"] = (
        HunyuanInstructLoaderInstantTensor
    )
    NODE_CLASS_MAPPINGS["HunyuanInstructLoaderFastSafetensors"] = (
        HunyuanInstructLoaderFastSafetensors
    )
    NODE_DISPLAY_NAME_MAPPINGS["HunyuanInstructLoaderInstantTensor"] = (
        "Hunyuan Instruct Loader (InstantTensor)"
    )
    NODE_DISPLAY_NAME_MAPPINGS["HunyuanInstructLoaderFastSafetensors"] = (
        "Hunyuan Instruct Loader (FastSafetensors)"
    )
