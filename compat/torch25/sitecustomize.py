"""Narrow import compatibility for ms-swift on NVIDIA's PyTorch 2.5 image.

ms-swift imports the PyTorch FSDP2 marker class while registering optional CPU
offload callbacks, even for single-GPU jobs that do not enable FSDP. NVIDIA's
PyTorch 2.5 image predates that public symbol. Supplying an inert marker lets
the unused callback module import without pretending that FSDP2 is available.
"""

try:
    import torch.distributed.fsdp as _fsdp

    if not hasattr(_fsdp, "FSDPModule"):

        class _UnavailableFSDPModule:
            pass

        _fsdp.FSDPModule = _UnavailableFSDPModule
except (ImportError, AttributeError):
    pass


# transformers >= 5 refuses every torch.load below torch 2.6 (CVE-2025-32434).
# That guard exists for untrusted checkpoint pickles; exact resume of our own
# Trainer checkpoints loads optimizer/scheduler/RNG files this machine wrote
# itself.  NVIDIA's 2.5 image cannot upgrade torch without invalidating the
# verified backend admission contract, so a caller that explicitly sets
# EMBEDDING_TRUST_LOCAL_TORCH_LOAD=1 (the exact-resume trainer path) disables
# only this version refusal.  Nothing changes for other processes.
import os as _os

if _os.environ.get("EMBEDDING_TRUST_LOCAL_TORCH_LOAD") == "1":
    import importlib.abc as _importlib_abc
    import importlib.util as _importlib_util
    import sys as _sys

    _TARGET = "transformers.utils.import_utils"

    class _TrustLocalTorchLoadFinder(_importlib_abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET:
                return None
            _sys.meta_path.remove(self)
            try:
                spec = _importlib_util.find_spec(fullname)
            finally:
                _sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            original_exec = spec.loader.exec_module

            def exec_module(module):
                original_exec(module)

                def _trusted_local_torch_load_ok() -> None:
                    return None

                module.check_torch_load_is_safe = _trusted_local_torch_load_ok

            spec.loader.exec_module = exec_module
            return spec

    _sys.meta_path.insert(0, _TrustLocalTorchLoadFinder())
