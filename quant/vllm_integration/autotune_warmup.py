"""Requests for vLLM's model-runner autotune warmup."""

_MXFP4_REQUESTED = False


def request_mxfp4_warmup() -> None:
    global _MXFP4_REQUESTED
    _MXFP4_REQUESTED = True


def mxfp4_warmup_requested() -> bool:
    from ..flashinfer_integration.mxfp4_fold_tuner import tuner_enabled

    return _MXFP4_REQUESTED and tuner_enabled()
