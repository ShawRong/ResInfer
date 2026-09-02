from __future__ import annotations


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def fmt_float(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:g}"


def safe_token(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "p")
        .replace("-", "m")
    )


def is_none_format(format_name: str) -> bool:
    return format_name == "NONE" or format_name.startswith("NONE(")
