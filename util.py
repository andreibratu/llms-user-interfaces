from typing import Any, Dict


def shorten_string_values(
        dict_val: Dict[str, Any], limit: int = 80
) -> Dict[str, Any]:
    for k in dict_val:
        s_reprs = str(dict_val[k])
        if len(s_reprs) > limit:
            dict_val[k] = str(dict_val[k])[:limit] + "<truncated>"
    return dict_val