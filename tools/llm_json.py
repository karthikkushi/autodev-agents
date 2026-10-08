"""
One JSON parser for model replies.

Models wrap JSON in ``` fences, put reasoning before it and prose after it,
leave trailing commas, use single quotes, or get cut off mid-array. Each
agent used to do its own `re.search(r"\\{.*\\}")` + json.loads: the greedy
match swallowed braces from prose after the JSON, and one small syntax slip
threw the whole reply away. (Code replies use FILE blocks instead — see
tools/file_blocks.py — this is only for the JSON ones.)
"""
import json
import re

from rich.console import Console

console = Console()

_FENCE_MARK = re.compile(r"```[\w+-]*")
_OPENER = {dict: "{", list: "["}
MAX_TEXT = 200_000
MAX_REPAIRS = 20


def parse_llm_json(text: str, expect: type = dict, accept=None):
    """The first JSON value of type `expect` (dict or list) in a model reply,
    or None if there is nothing usable.

    1. ``` fences are stripped;
    2. strict parse: json.JSONDecoder().raw_decode from each "{" / "[" in
       turn, so prose before and after is ignored and braces inside strings
       don't matter (the approach of tools/file_blocks.parse_meta_json);
    3. json_repair only as a last resort — trailing commas, single quotes, a
       reply cut off mid-array — with a warning, because a repaired reply may
       be missing its end.

    A non-empty value wins over an empty one. `accept` lets a caller skip
    values that parse but aren't what it wants (e.g. a stray [1] in the
    planner's reasoning before the real task list)."""
    if expect not in _OPENER:
        raise ValueError("expect must be dict or list")
    s = _FENCE_MARK.sub("", text or "")[:MAX_TEXT]
    wanted = (lambda v: isinstance(v, expect) and (accept is None or accept(v)))
    starts = [m.start() for m in re.finditer(r"[\[{]", s)]

    decoder, empty = json.JSONDecoder(), None
    for i in starts:
        try:
            value, _ = decoder.raw_decode(s, i)
        except ValueError:
            continue
        if wanted(value):
            if value:
                return value
            if empty is None:
                empty = value

    try:
        from json_repair import repair_json
    except ImportError:  # not installed: behave like a strict parser
        return empty
    for i in [i for i in starts if s[i] == _OPENER[expect]][:MAX_REPAIRS]:
        try:
            value = repair_json(s[i:], return_objects=True)
        except Exception:
            continue
        if wanted(value) and value:
            console.print("[yellow]⚠️  A model reply's JSON needed repair — it may have been cut off[/yellow]")
            return value
    return empty
