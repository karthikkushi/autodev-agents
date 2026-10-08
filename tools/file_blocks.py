"""
File blocks — how every agent that writes files gets them back from a model.

Models write measurably worse code when it has to be escaped inside JSON
(aider's benchmark: every model tested scored lower, DeepSeek worst — quotes
and newlines get mangled), and a JSON reply cut off by max_tokens loses every
file in it. Plain blocks need no escaping, and a reply that gets cut off still
yields every file that finished before the cut.

    === FILE: app.py ===
    <file content, exactly as saved>
    === END FILE ===
"""
import json
import re

FILE_FORMAT = """Return every file as a block exactly like this — not JSON, nothing escaped:

=== FILE: path/to/file.ext ===
<the complete file content, exactly as it should be saved>
=== END FILE ===

One block per file. Paths are relative to {root}."""

# The closing "===" on both marker lines is optional: models regularly write
# "=== FILE: src/app.py" without it, and a strict match once threw away a
# correct fix (and with it, the decision agent's whole answer).
_BLOCK = re.compile(
    r"^={3,}[ \t]*FILE:?[ \t]*(?P<path>[^\n=]+?)[ \t]*(?:={3,})?[ \t]*\n"
    r"(?P<body>.*?)"
    r"^={3,}[ \t]*END(?:[ \t]+OF)?[ \t]+FILE[ \t]*(?:={3,})?[ \t]*$",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
_HEADER = re.compile(r"^={3,}[ \t]*FILE:?[ \t]*[^\n=]+?[ \t]*(?:={3,})?[ \t]*$", re.MULTILINE | re.IGNORECASE)
_FENCE = re.compile(r"\A[ \t]*```[\w+.#-]*[ \t]*\n(?P<code>.*?)\n?[ \t]*```[ \t]*\Z", re.DOTALL)

# "Lazy" replies that would overwrite a real file with a stub: the format's
# own placeholder copied back ("<the entire file content>" wiped a 37-test
# suite), or an elision comment standing in for code the model skipped.
# A portfolio's <main> was emptied twice by comments without a leading "...":
# "<!-- Sections omitted for brevity -->", "<!-- Sections content remains identical -->".
_PLACEHOLDER = re.compile(r"\A\s*<[^<>\n]*\b(content|file|code|here)\b[^<>\n]*>\s*\Z", re.IGNORECASE)
# The wording rules only match at the start of a comment, where the comment
# *is* the elision; "Keep existing items unchanged" is a real comment.
_LEAD = r"(?:\.\.\.|…|\()?[ \t]*(?:the\s+)?"
_ELIDED_NOUN = (r"(?:sections?|content|code|markup|html|css|styles?|rules|functions?|methods?|logic|implementation"
                r"|rest|remainder|body|details|everything\s+else|all\s+else|(?:other|remaining|existing|previous)\s+\w+)")
_ELISION = re.compile(
    r"^[ \t]*(?:\#|//|/\*|<!--|\{/\*)[ \t]*(?:"
    r"(?:\.\.\.|…)[ \t]*(?:rest|existing|remaining|previous|same|other|unchanged)\b"  # "// ... rest of code"
    r"|(?:\.\.\.|…)[ \t]*(?:-->|\*/\}?)?[ \t]*$"  # a comment holding only "..."
    rf"|{_LEAD}for\s+brevity\b"
    r"|[^\n]*?\b(?:omitted|elided|skipped|removed|shortened|not\s+shown|left\s+out)\s+(?:here\s+)?for\s+brevity\b"
    r"|[^\n]*?\(\s*(?:omitted|elided)\s*\)"
    rf"|{_LEAD}{_ELIDED_NOUN}\b[^\n]{{0,20}}?\b(?:omitted|elided)\b"
    rf"|{_LEAD}{_ELIDED_NOUN}(?:\s+\w+){{0,3}}?\s+(?:(?:remains?|stays?)\s+(?:unchanged|identical|the\s+same|as\s+(?:before|is))"
    r"|(?:(?:is|are|was|kept|left)\s+)?(?:unchanged|identical|as\s+before))\b"
    rf"|{_LEAD}(?:add\s+)?(?:other|remaining|more|additional)\s+\w+(?:\s+\w+)?\s+(?:go(?:es)?\s+)?here[ \t.!]*(?:-->|\*/\}}?)?[ \t]*$"
    r")",
    re.MULTILINE | re.IGNORECASE,
)
# A bare "..." standing in for content: a design fix returned
# <section id="hero">...</section> for four sections and emptied a portfolio.
_ELLIPSIS_LINE = re.compile(r"^[ \t]*(?:\.\.\.|…)[ \t]*$", re.MULTILINE)
_ELLIPSIS_CONTAINER = re.compile(
    r"<(section|header|main|footer|nav|article|aside|div|ul|ol|form|body|head)\b[^>]*>\s*(?:\.\.\.|…)\s*</\1\s*>",
    re.IGNORECASE)


def is_stub(body: str, path: str = "") -> bool:
    """True if a block body is a placeholder or has content elided. Python is
    left out of the "..." rules: there a bare `...` is a real statement."""
    if _PLACEHOLDER.match(body) or _ELISION.search(body):
        return True
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext in ("html", "htm", "jinja", "j2"):
        return bool(_ELLIPSIS_CONTAINER.search(body) or len(_ELLIPSIS_LINE.findall(body)) >= 2)
    if ext in ("js", "mjs", "ts", "css"):
        return bool(_ELLIPSIS_LINE.search(body))  # never valid there
    return False


def parse_file_blocks(text: str) -> dict:
    """{path: content} for every complete block. A block cut off before its END
    marker is dropped; the complete ones before it are kept. So is a stub
    (see is_stub) — writing it would destroy the file it names."""
    files = {}
    for m in _BLOCK.finditer(text or ""):
        path = m.group("path").strip().strip("`*'\"")
        body = m.group("body")
        fenced = _FENCE.match(body.strip("\n"))
        if fenced:  # the model wrapped the content in ``` anyway
            body = fenced.group("code") + "\n"
        if is_stub(body, path):
            continue
        files[path] = body
    return files


def was_cut_off(text: str) -> bool:
    """True if a FILE block was started but never finished (reply truncated)."""
    return len(_HEADER.findall(text or "")) > len(_BLOCK.findall(text or ""))


def parse_files(text: str):
    """File blocks first; falls back to the old {path: content} JSON so a model
    that ignores the format still works. None if neither is present."""
    files = parse_file_blocks(text)
    if files:
        return files
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
        except Exception:
            return None
        if isinstance(data, dict) and data and all(isinstance(v, str) for v in data.values()):
            return data
    return None


EDIT_FORMAT = """Return only the lines that change, as edit blocks (not the whole file):

=== EDIT: {path} ===
<<<<<<< SEARCH
exact lines as they are in the current file (enough of them to be unique)
=======
the lines that replace them
>>>>>>> REPLACE
=== END EDIT ===

SEARCH must match the current file exactly, including indentation. For several changes, put
several SEARCH/REPLACE pairs inside the one EDIT block."""

_EDIT_BLOCK = re.compile(
    r"^={3,}[ \t]*EDIT:?[ \t]*(?P<path>[^\n=]+?)[ \t]*(?:={3,})?[ \t]*\n(?P<body>.*?)"
    r"^={3,}[ \t]*END[ \t]+EDIT[ \t]*(?:={3,})?[ \t]*$",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
_SEARCH_REPLACE = re.compile(
    r"^<{5,9}[ \t]*SEARCH[ \t]*\n(?P<search>.*?)^={5,9}[ \t]*\n(?P<replace>.*?)^>{5,9}[ \t]*REPLACE[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def parse_edits(text: str) -> dict:
    """{path: [(search, replace), ...]} from EDIT blocks."""
    edits = {}
    for m in _EDIT_BLOCK.finditer(text or ""):
        pairs = [(sr.group("search"), sr.group("replace")) for sr in _SEARCH_REPLACE.finditer(m.group("body"))]
        if pairs:
            edits.setdefault(m.group("path").strip().strip("`*'\""), []).extend(pairs)
    return edits


def apply_edits(original: str, pairs: list) -> tuple[str, int]:
    """Apply (search, replace) pairs in order. Exact match first; then a match
    that ignores trailing whitespace per line (models often drop it).
    Returns (new text, number of pairs that didn't match)."""
    text, failed = original, 0
    for search, replace in pairs:
        if search and search in text:
            text = text.replace(search, replace, 1)
            continue
        want = [line.rstrip() for line in search.strip("\n").splitlines()]
        lines = text.splitlines(keepends=True)
        for i in range(len(lines) - len(want) + 1):
            if want and [l.rstrip() for l in lines[i:i + len(want)]] == want:
                text = "".join(lines[:i]) + replace + "".join(lines[i + len(want):])
                break
        else:
            failed += 1
    return text, failed


def parse_meta_json(text: str) -> dict:
    """The JSON object an agent writes *before* its file blocks (findings,
    decision, notes). Searching only that part keeps braces inside the code
    from confusing the match. {} if there isn't a valid one."""
    head = _HEADER.split(text or "", maxsplit=1)[0]
    # Decode from each "{" in turn rather than one greedy {.*} match, which
    # swallowed braces from code after the JSON and failed to parse.
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", head):
        try:
            data, _ = decoder.raw_decode(head, m.start())
        except ValueError:
            continue
        if isinstance(data, dict) and data:
            return data
    return {}
