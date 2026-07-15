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
