from typing import Any


def shorten_string_values(dict_val: dict[str, Any], limit: int = 80) -> dict[str, Any]:
    for k in dict_val:
        string_representation = str(dict_val[k])
        if len(string_representation) > limit:
            dict_val[k] = str(dict_val[k])[:limit] + "<truncated>"
    return dict_val
