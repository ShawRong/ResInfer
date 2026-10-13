"""Narrow repair for lighteval 0.13's LiteLLM -> vLLM parameter transport."""


def install(generation_parameters):
    original = generation_parameters.to_litellm_dict
    if getattr(original, "_resinfer_sampling_compat", False):
        return

    def to_litellm_dict(self):
        params = original(self)
        # OpenAI-native parameter, omitted by lighteval 0.13.
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        # vLLM extensions go through LiteLLM's official extra_body passthrough;
        # top-level non-OpenAI fields may be dropped with drop_params=True.
        extra = dict(params.get("extra_body") or {})
        for name in ("top_k", "min_p", "repetition_penalty"):
            value = getattr(self, name)
            params.pop(name, None)
            if value is not None:
                extra[name] = value
        if extra:
            params["extra_body"] = extra
        return params

    to_litellm_dict._resinfer_sampling_compat = True
    generation_parameters.to_litellm_dict = to_litellm_dict
