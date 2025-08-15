import json


class CompactJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that formats bottom-level objects on a single line
    and higher-level objects with indentation.
    """
    def __init__(self, *args, **kwargs):
        # Extract indent, and if it"s missing or None, default to 4
        indent = kwargs.pop("indent", None)
        if indent is None:
            indent = 4
        kwargs["indent"] = indent

        super().__init__(*args, **kwargs)

    def iterencode(self, o, _one_shot=False):
        def _iterencode(obj, level):
            # bottom-level dict: all values are primitives
            if isinstance(obj, dict) and level > 0 and all(
                not isinstance(v, (dict, list)) for v in obj.values()
            ):
                yield json.dumps(obj,
                                 separators=(",", ": "),
                                 ensure_ascii=self.ensure_ascii)
                return

            # bottom-level list: all items are primitives
            if isinstance(obj, list) and level > 0 and all(
                not isinstance(v, (dict, list)) for v in obj
            ):
                yield json.dumps(obj,
                                 separators=(",", ":"),
                                 ensure_ascii=self.ensure_ascii)
                return

            # pretty-print dict
            if isinstance(obj, dict):
                yield "{\n"
                indent_str = " " * (self.indent * (level + 1))
                for i, (k, v) in enumerate(obj.items()):
                    if i:
                        yield ",\n"
                    yield indent_str + json.dumps(k) + ": "
                    yield from _iterencode(v, level + 1)
                yield "\n" + " " * (self.indent * level) + "}"
                return

            # pretty-print list
            if isinstance(obj, list):
                yield "[\n"
                indent_str = " " * (self.indent * (level + 1))
                for i, v in enumerate(obj):
                    if i:
                        yield ",\n"
                    yield indent_str
                    yield from _iterencode(v, level + 1)
                yield "\n" + " " * (self.indent * level) + "]"
                return

            # primitive fallback
            yield json.dumps(obj, ensure_ascii=self.ensure_ascii)

        return _iterencode(o, 0)


def json_to_string(data):
    """
    Convert JSON data to a formatted string with bottom-level objects on a single line.
    
    Args:
        data: The JSON data to format
        
    Returns:
        A formatted JSON string
    """
    return json.dumps(data, cls=CompactJSONEncoder)
