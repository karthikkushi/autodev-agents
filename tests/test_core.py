"""Offline checks for the pipeline's parsing and loop rules — no model calls.

    python3 -m pytest tests/ -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.file_blocks import (apply_edits, parse_edits, parse_file_blocks, parse_files,
                               parse_meta_json, was_cut_off)
from tools.file_ops import python_syntax_error
from agents.tester_debugger import head_tail, pytest_counts, _error_query
from agents.decision import _apply_decision

# The reply that sent tip_calculator back to the planner on 2026-09-30: no
# closing "===" on the marker lines, and code braces after the JSON.
DECISION_REPLY = '''{"decision": "fix_code", "reasoning": "Use ROUND_HALF_UP for money.", "confidence": 0.95}

=== FILE: src/calculator.py
from decimal import Decimal, ROUND_HALF_UP

def money(x):
    return {"value": float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))}
=== END FILE
'''


def test_unclosed_markers_are_read():
    files = parse_file_blocks(DECISION_REPLY)
    assert list(files) == ["src/calculator.py"]
    assert "ROUND_HALF_UP" in files["src/calculator.py"]


def test_meta_json_ignores_code_braces():
    assert parse_meta_json(DECISION_REPLY)["decision"] == "fix_code"
    assert parse_meta_json('x {bad} {"notes": ["a"]} {y}') == {"notes": ["a"]}
    assert parse_meta_json("no json here") == {}


def test_closed_markers_and_fences():
    text = '=== FILE: app.py ===\n```python\nx = {"a": 1}\n```\n=== END FILE ===\n'
    assert parse_file_blocks(text) == {"app.py": 'x = {"a": 1}\n'}


def test_cut_off_reply_keeps_finished_files():
    text = "=== FILE: a.py ===\na = 1\n=== END FILE ===\n=== FILE: b.py ===\nb = "
    assert list(parse_file_blocks(text)) == ["a.py"]
    assert was_cut_off(text)


def test_stub_blocks_are_dropped():
    # Nemotron copied the format's placeholder back and wiped a 37-test file.
    assert parse_file_blocks("=== FILE: tests/test_calc.py ===\n<the entire file content>\n=== END FILE ===\n") == {}
    lazy = "=== FILE: app.py ===\nimport os\n# ... rest of the code unchanged\n=== END FILE ===\n"
    assert parse_file_blocks(lazy) == {}
    real = "=== FILE: page.html ===\n<!-- header -->\n<main><p>Hello</p></main>\n=== END FILE ===\n"
    assert "page.html" in parse_file_blocks(real)   # an ordinary comment isn't an elision


def test_worded_elisions_are_dropped():
    # The first two emptied a portfolio's <main>: a review fix, then a security patch.
    for comment in ("<!-- Sections omitted for brevity -->", "<!-- Sections content remains identical -->",
                    "<!-- Other sections go here -->", "<!-- ... -->"):
        lazy = f"=== FILE: src/index.html ===\n<main>\n  {comment}\n</main>\n=== END FILE ===\n"
        assert parse_file_blocks(lazy) == {}, comment
    for comment in ("// Keep existing items unchanged when merging", "// Error details omitted from the response",
                    "/* Same as above but for dark mode */", "// Hide the rest of the page while the modal is open"):
        real = f"=== FILE: src/js/main.js ===\n{comment}\nrun();\n=== END FILE ===\n"
        assert "src/js/main.js" in parse_file_blocks(real), comment


def test_a_review_fix_cannot_drop_sections(tmp_path, monkeypatch):
    # A fix adding SEO tags rewrote the page without the sections the fix
    # before it had just restored.
    import agents.coder as coder
    src = tmp_path / "src"
    src.mkdir()
    page = "".join(f'<section id="s{i}"><h2>S{i}</h2><p>text</p></section>\n' for i in range(4))
    (src / "index.html").write_text(page)
    prompts = []

    class FakeLLM:   # the page back with one section left, and no usable edits when asked again
        def invoke(self, messages):
            prompts.append(messages[0].content)
            return type("R", (), {"content": "=== FILE: src/index.html ===\n<title>Asha</title>\n"
                                             '<section id="s0"><h2>S0</h2></section>\n=== END FILE ===\n'})()

    class NoMemory:
        def __init__(self, *a): pass
        def search(self, *a): return ""
        def get_failures(self): return []
        def store(self, *a, **k): pass

    monkeypatch.setattr(coder.router, "get_llm", lambda *a, **k: FakeLLM())
    monkeypatch.setattr(coder, "AgentMemory", NoMemory)
    task = {"id": 9000, "name": "Fix: missing SEO tags", "description": "In src/index.html: add a <title>.",
            "file": "src/index.html", "type": "fix"}
    state = dict(task_list=[task], current_task_index=0, project_path=str(tmp_path), project_name="p",
                 project_type="webapp", architecture="", research_notes="", design_content="Asha, digital marketer",
                 errors=[], files_written=[], agent_logs=[], coder_round=1)
    coder.run_coder(state)
    assert (src / "index.html").read_text() == page
    assert any("kept the old file" in e for e in state["errors"])
    assert "Asha, digital marketer" in prompts[0]   # web fixes get the design's content to work from


def test_python_syntax_guard():
    assert python_syntax_error("a.py", "<the entire file content>\n").startswith("SyntaxError")
    assert python_syntax_error("a.py", "x = 1\n") == ""
    assert python_syntax_error("a.css", "not { python") == ""


def test_json_fallback():
    assert parse_files('{"main.py": "print(1)\\n"}') == {"main.py": "print(1)\n"}
    assert parse_files("nothing") is None


def test_edits_with_and_without_closing_markers():
    for close in ("===", ""):
        text = f"=== EDIT: a.css {close}\n<<<<<<< SEARCH\nred\n=======\nblue\n>>>>>>> REPLACE\n=== END EDIT {close}\n"
        pairs = parse_edits(text)["a.css"]
        assert apply_edits("red\n", pairs) == ("blue\n", 0)


def test_edit_tolerates_trailing_whitespace():
    assert apply_edits("a  \nb\n", [("a\nb\n", "c\n")]) == ("c\n", 0)
    assert apply_edits("a\n", [("zzz\n", "c\n")]) == ("a\n", 1)


def test_pytest_summary_and_trimming():
    out = "." * 500 + "F\nE   assert 0.01 == 0.0\n" + "x" * 4000 + "\n1 failed, 16 passed in 0.3s\n"
    assert pytest_counts(out) == (16, 1)
    assert pytest_counts("2 failed, 3 passed, 1 error in 1s") == (3, 3)
    trimmed = head_tail(out, 3000)
    assert len(trimmed) < 3010 and "1 failed, 16 passed" in trimmed
    assert _error_query(out) == "assert 0.01 == 0.0"


def _state(tmp_path, **kw):
    (tmp_path / "src").mkdir(exist_ok=True)
    s = dict(project_path=str(tmp_path), errors=["err"], coder_round=0, decision_count=0,
             backtrack_count=0, task_list=[1, 2], current_task_index=2, known_issues=[],
             files_written=["a"], plan="p")
    s.update(kw)
    return s


def test_near_green_build_is_never_replanned(tmp_path):
    s = _state(tmp_path)
    _apply_decision(s, {"decision": "change_approach"}, fix_only=True)
    assert s["phase"] == "finalizing" and s["backtrack_count"] == 0
    assert s["task_list"] == [1, 2] and s["known_issues"] == ["err"]


def test_near_green_fix_is_applied_and_retested(tmp_path):
    s = _state(tmp_path)
    _apply_decision(s, {"decision": "fix_code", "code_fixes": {"src/calc.py": "x = 1\n"}}, fix_only=True)
    assert s["phase"] == "testing"
    assert (tmp_path / "src" / "calc.py").read_text() == "x = 1\n"


def test_normal_change_approach_backtracks(tmp_path):
    s = _state(tmp_path)
    _apply_decision(s, {"decision": "change_approach"})
    assert s["phase"] == "planning" and s["backtrack_count"] == 1 and s["task_list"] == []


def test_decision_limit_stops_the_loop(tmp_path):
    s = _state(tmp_path, decision_count=2)
    _apply_decision(s, {"decision": "change_approach"})
    assert s["phase"] == "finalizing"


def _ui_project(tmp_path, monkeypatch, scores, fix_css, tests_pass=True, page_problems=None):
    import agents.ui_review as ui
    static = tmp_path / "src" / "static"
    static.mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (static / "style.css").write_text("good\n")
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"png")
    reviews = iter(scores)

    class FakeLLM:
        def invoke(self, _):
            return type("R", (), {"content": f"=== FILE: src/static/style.css ===\n{fix_css}=== END FILE ===\n"
                                             "=== FILE: src/static/extra.css ===\nx\n=== END FILE ===\n"})()

    monkeypatch.setattr(ui.webapp, "browser", lambda: "chrome")
    monkeypatch.setattr(ui, "_shoot", lambda p, tag: {"desktop": str(shot)})
    monkeypatch.setattr(ui, "_critique", lambda s, shots: {"score": next(reviews), "issues": [{"issue": "x"}]})
    monkeypatch.setattr(ui, "_tests_pass", tests_pass if callable(tests_pass) else (lambda p: tests_pass))
    monkeypatch.setattr(ui, "_failing_tests", lambda p: "E   assert '#bill' in page")
    monkeypatch.setattr(ui, "_page_problems", page_problems or (lambda p: set()))
    monkeypatch.setattr(ui.router, "get_llm", lambda *a, **k: FakeLLM())
    state = dict(project_type="webapp", project_path=str(tmp_path), ui_spec="", known_issues=[], agent_logs=[])
    return ui.run_ui_review(state), static


def test_visual_qa_rolls_back_a_worse_fix(tmp_path, monkeypatch):
    state, static = _ui_project(tmp_path, monkeypatch, [6.0, 4.0], "bad\n")
    assert (static / "style.css").read_text() == "good\n"
    assert not (static / "extra.css").exists()
    assert state["ui_review_score"] == 6.0


def test_visual_qa_keeps_a_better_fix(tmp_path, monkeypatch):
    state, static = _ui_project(tmp_path, monkeypatch, [6.0, 8.0], "better\n")
    assert (static / "style.css").read_text() == "better\n"
    assert state["ui_review_score"] == 8.0


def test_cut_off_replies_are_kept(tmp_path, monkeypatch):
    import router.llm_router as r
    monkeypatch.setattr(r, "CUT_OFF_DIR", tmp_path)
    monkeypatch.setattr(r, "CUT_OFF_KEEP", 2)
    for i in range(3):
        (tmp_path / f"2026010{i}-000000_x_m.txt").write_text("old")
    r._keep_cut_off_reply("nvidia_nim/nvidia/model-a", "coding", "reply")
    kept = sorted(p.name for p in tmp_path.glob("*.txt"))
    assert len(kept) == 2 and kept[-1].endswith("_coding_model-a.txt")
    fake = type("R", (), {"choices": [type("C", (), {"finish_reason": "length"})()]})()
    assert r._finish_reason(fake) == "length" and r._finish_reason(None) == ""


def _first_pixel(png: bytes) -> tuple:
    """RGB of the top-left pixel (always stored unfiltered in a PNG)."""
    import struct
    import zlib
    pos, idat = 8, b""
    while pos < len(png):
        length, kind = struct.unpack(">I4s", png[pos:pos + 8])
        if kind == b"IDAT":
            idat += png[pos + 8:pos + 8 + length]
        pos += 12 + length
    return tuple(zlib.decompress(idat)[1:4])


def test_phone_screenshot_has_a_real_phone_viewport(tmp_path):
    # Chrome's --window-size can't go under 500px on macOS: "390px" shots were
    # a 500px layout cropped. The page is green only if the viewport is <= 390.
    import pytest
    from tools import webapp
    if not webapp.browser():
        pytest.skip("no Chrome/Edge installed")
    page = tmp_path / "w.html"
    page.write_text('<meta name="viewport" content="width=device-width, initial-scale=1">'
                    "<style>body{margin:0;height:100vh;background:#00ff00}"
                    "@media (min-width: 391px){body{background:#ff0000}}</style>")
    out = tmp_path / "w.png"
    assert webapp.screenshot(page.as_uri(), str(out), 390, 844)
    r, g, b = _first_pixel(out.read_bytes())
    assert g > 200 and r < 50, f"viewport wider than 390px (pixel {r},{g},{b})"


def test_wrapped_quota_error_counts_as_rate_limit():
    import router.llm_router as r
    BadRequestError = type("BadRequestError", (Exception,), {})
    wrapped = BadRequestError('GeminiException BadRequestError - {\n  "error": {\n    "code": 429,')
    r._rate_backoff.pop("m", None)
    assert r._cooldown_seconds("m", wrapped) == r._COOLDOWN["rate"]
    assert r._cooldown_seconds("m2", BadRequestError("bad prompt")) == r._COOLDOWN["other"]


def test_visual_qa_keeps_a_fix_when_tests_were_already_failing(tmp_path, monkeypatch):
    # Tests that failed before the fix can't blame it: the Playwright suite
    # of a static site errored every time and would have undone every fix.
    state, static = _ui_project(tmp_path, monkeypatch, [6.0, 8.0], "better\n", tests_pass=False)
    assert (static / "style.css").read_text() == "better\n"
    assert state["ui_review_score"] == 8.0


def test_visual_qa_gets_one_more_try_when_a_fix_breaks_the_tests(tmp_path, monkeypatch):
    # Passing before, broken by the design fix, passing again after the retry
    # that showed the fixer the failing test: the design fix is kept.
    results = iter([True, False, True])
    state, static = _ui_project(tmp_path, monkeypatch, [6.0, 8.0], "better\n",
                                tests_pass=lambda p: next(results, True))
    assert (static / "style.css").read_text() == "better\n"
    assert state["ui_review_score"] == 8.0


def test_visual_qa_undoes_a_fix_that_empties_the_page(tmp_path, monkeypatch):
    # The page check before the fix is clean, after it a section is empty.
    checks = iter([set(), {"empty-section: section #hero shows nothing"}])
    state, static = _ui_project(tmp_path, monkeypatch, [6.0, 8.0], "bad\n",
                                page_problems=lambda p: next(checks, set()))
    assert (static / "style.css").read_text() == "good\n"
    assert state["ui_review_score"] == 6.0


def test_a_static_site_is_never_run_with_npm(tmp_path):
    # A stray package.json sent all four debug attempts to fixing `npm test`.
    from agents.tester_debugger import _detect_entry_point
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "index.html").write_text("<h1>Asha</h1>\n")
    (tmp_path / "package.json").write_text('{"name": "portfolio"}\n')
    cmd, cwd, _ = _detect_entry_point(str(tmp_path), "fullstack")
    assert "npm" not in cmd and "HTMLParser" in cmd and cwd == str(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_page.py").write_text("def test_x(): pass\n")
    assert "-m pytest tests/" in _detect_entry_point(str(tmp_path), "fullstack")[0]   # real tests come first


def test_visual_qa_never_keeps_a_fix_that_deletes_sections(tmp_path, monkeypatch):
    # Shown 8,000 characters of a 25 KB page, the fixer returned its hero as
    # the whole page — and the design score, which sees one screen, went up.
    import agents.ui_review as ui
    src = tmp_path / "src"
    src.mkdir()
    (tmp_path / "docs").mkdir()
    page = "".join(f'<section id="s{i}"><h2>S{i}</h2><p>{"text " * 400}</p></section>\n' for i in range(6))
    (src / "index.html").write_text(page)
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"png")
    prompts, reviews = [], iter([4.0, 5.0])

    class FakeLLM:
        def invoke(self, messages):
            prompts.append(str(messages))
            return type("R", (), {"content": '=== FILE: src/index.html ===\n<section id="s0"><h2>S0</h2>'
                                             "</section>\n=== END FILE ===\n"})()

    monkeypatch.setattr(ui.webapp, "browser", lambda: "chrome")
    monkeypatch.setattr(ui, "_shoot", lambda p, tag: {"desktop": str(shot)})
    monkeypatch.setattr(ui, "_critique", lambda s, shots: {"score": next(reviews), "issues": [{"issue": "x"}]})
    monkeypatch.setattr(ui, "_tests_pass", lambda p: True)
    monkeypatch.setattr(ui, "_page_problems", lambda p: set())
    monkeypatch.setattr(ui.router, "get_llm", lambda *a, **k: FakeLLM())
    state = dict(project_type="webapp", project_path=str(tmp_path), project_name="p", ui_spec="",
                 known_issues=[], agent_logs=[])
    ui.run_ui_review(state)
    assert (src / "index.html").read_text() == page
    assert '<section id="s5">' in prompts[0]   # the fixer is shown the whole page


def test_review_loop_keeps_going_while_it_helps():
    from agents.reviewer import _stalled
    report = lambda highs: "## Issues\n" + "- [HIGH] app.py: x\n" * highs
    rounds = lambda *counts: [report(c) for c in counts]
    assert not _stalled([], 5)                                   # first round
    assert not _stalled(rounds(4, 4), 9)                         # build 6: new problems found, keep going
    assert not _stalled(rounds(6, 4, 5), 5)                      # round 2 set a new best
    assert not _stalled(rounds(9, 7, 5), 3)                      # steady progress
    assert _stalled(rounds(4, 4, 9), 5)                          # three rounds without beating 4
    assert _stalled(rounds(6, 4, 5, 5), 5)                       # stuck above the best of 4