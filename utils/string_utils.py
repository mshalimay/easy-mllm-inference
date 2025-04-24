import re
from typing import Any


def safe_format(string_template: str, fill_with: str = "", **kwargs: Any) -> str:
    """
    Formats a given template using the provided keyword arguments.
    Missing keys in the template are replaced with an empty string.

    Args:
        template (str): The string template with placeholders.
        **kwargs: Key-value pairs for formatting.

    Returns:
        str: The formatted string with missing keys as empty strings.
    """

    class DefaultDict(dict[Any, Any]):
        def __missing__(self, key: Any) -> Any:
            return fill_with

    return string_template.format_map(DefaultDict(**kwargs))


def clean_spaces(text: str) -> str:
    """Replace multiple newlines with a single newline and trim excess whitespace."""
    text = re.sub(r"\n+", "\n", text)
    return re.sub(r"[ \t\r]+", " ", text).strip()
