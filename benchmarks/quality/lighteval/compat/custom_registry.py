"""Opt-in custom-only discovery for isolated offline lighteval 0.13 tasks."""


def install(registry):
    def load_custom_only(custom_tasks=None, load_multilingual=False):
        if custom_tasks is None or load_multilingual:
            raise ValueError("Custom-only discovery requires custom_tasks and no multilingual tasks")
        module = registry.create_custom_tasks_module(custom_tasks)
        configs = registry._extract_configs(module)
        if not configs:
            raise ValueError("Custom task module exports no task configs")
        return configs

    registry.load_all_task_configs = staticmethod(load_custom_only)
