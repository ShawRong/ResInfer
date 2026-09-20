# SPDX-FileCopyrightText: Copyright (c) 2025 ResInfer contributors
# SPDX-License-Identifier: Apache-2.0
"""Production phase-separated ResAttn backend with the stock output ABI."""
from functools import partialmethod
from .attn_backend import ResAttnImpl
from .attn_backend_flashinfer import ResAttnFlashInferBackend, ResAttnFlashInferImpl


class ResAttnDirectOutputImpl(ResAttnFlashInferImpl):
    _decode_forward = partialmethod(ResAttnImpl._decode_forward, direct_output=True)


class ResAttnDirectOutputBackend(ResAttnFlashInferBackend):
    @staticmethod
    def get_impl_cls():
        return ResAttnDirectOutputImpl
