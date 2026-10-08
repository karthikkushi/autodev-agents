"""The shared JSON parser and the free static checks — offline, no model calls."""
import pytest

from tools import static_checks as sc
from tools.llm_json import parse_llm_json
from agents.planner import _parse_task_list


# ── parse_llm_json ───────────────────────────────────────────────────────────

def test_fenced_json():
    assert parse_llm_json('Here it is:\n```json\n{"pass": true, "gaps": []}\n```\nDone.') == {"pass": True, "gaps": []}


def test_prose_before_and_after():
    reply = 'Sure! {"score": 0.9, "issues": ["x"]} Hope that helps — {not json}'
    assert parse_llm_json(reply) == {"score": 0.9, "issues": ["x"]}


def test_trailing_comma_and_single_quotes_are_repaired():
    assert parse_llm_json('{"verdict": "pass", "issues": [],}') == {"verdict": "pass", "issues": []}
    assert parse_llm_json("{'project_type': 'webapp'}") == {"project_type": "webapp"}


def test_braces_inside_strings():
    assert parse_llm_json('{"reason": "a { brace } and ]", "pass": false}') == {"reason": "a { brace } and ]",
                                                                               "pass": False}


def test_reply_cut_off_mid_array_keeps_the_complete_items():
    tasks = parse_llm_json('[{"id": 1, "name": "A"}, {"id": 2, "name": "B", "desc', list)
    assert [t["name"] for t in tasks] == ["A", "B"]


def test_planner_reasoning_before_the_array():
    reply = ('Let me think. Step [1] is the model, then [see note]. Tasks:\n'
             '[{"id": 1, "name": "Create calculator"}, {"id": 2, "name": "Add UI"}]')
    tasks = _parse_task_list(reply)
    assert [t["name"] for t in tasks] == ["Create calculator", "Add UI"]
    assert tasks[0]["description"] == "Create calculator"   # the planner's default is kept
    assert _parse_task_list("no tasks here") is None


def test_nothing_usable_and_empty_values():
    assert parse_llm_json("no json at all") is None
    assert parse_llm_json("libraries: []", list) == []
    assert parse_llm_json('{"tasks": [{"name": "x"}]}', list) == [{"name": "x"}]


# ── static checks ────────────────────────────────────────────────────────────

@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "MYPY_CACHE", tmp_path / "mypy_cache")
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "src" / "mod.py").write_text("def present():\n    return 1\n")
    (root / "src" / "app.py").write_text(
        "import subprocess\n"
        "import nonexistent_module_xyz\n"
        "from mod import Missing\n\n\n"
        "def run(cmd):\n"
        "    subprocess.call(cmd, shell=True)\n"
        "    return undefined_name + 1\n")
    return root


def test_each_kind_of_bug_is_reported(project):
    code = sc.code_checks(str(project))
    security = sc.security_checks(str(project))
    found = {(f["tool"], f["code"], f["file"], f["line"]) for f in code["ruff"] + code["mypy"] + security["bandit"]}
    assert ("ruff", "F821", "src/app.py", 8) in found                      # undefined name
    assert ("mypy", "import-not-found", "src/app.py", 2) in found          # module that doesn't exist
    assert ("mypy", "attr-defined", "src/app.py", 3) in found              # from mod import Missing
    assert ("bandit", "B602", "src/app.py", 7) in found                    # subprocess with shell=True
    # mypy's copy of the undefined name is dropped (ruff already has it).
    assert not any(f["code"] == "name-defined" for f in code["mypy"])
    assert all(f["severity"] == "HIGH" for f in code["ruff"] + code["mypy"])
    # No venv in this project: pip-audit is skipped, not an error.
    assert security["pip-audit"] == [] and "no .venv" in sc._last_skip["pip-audit"]


def test_report_lists_every_tool(project):
    sc.write_report(str(project), sc.code_checks(str(project)))
    sc.write_report(str(project), {"bandit": sc.bandit_check(str(project))})
    page = (project / "docs" / "05c_static_checks.md").read_text()
    assert "## ruff" in page and "## mypy" in page and "## bandit" in page and "B602" in page


def test_a_syntax_error_is_reported_and_mypy_waits(project):
    (project / "src" / "broken.py").write_text("def oops(:\n    pass\n")
    code = sc.code_checks(str(project))
    assert any(f["code"] == "invalid-syntax" and f["file"] == "src/broken.py" for f in code["ruff"])
    assert code["mypy"] == [] and "syntax error" in sc._last_skip["mypy"]


def test_a_missing_tool_is_skipped_without_an_exception(project, monkeypatch):
    monkeypatch.setitem(sc._MODULES, "ruff", "no_such_tool_xyz")
    monkeypatch.setitem(sc._MODULES, "bandit", "no_such_tool_xyz")
    assert sc.ruff_check(["src/app.py"], root=project) == []
    assert sc.bandit_check(str(project)) == []
    assert "not installed" in sc._last_skip["ruff"]


def test_coder_syntax_check_includes_undefined_names(project):
    from agents.coder import _syntax_errors
    errors = _syntax_errors([str(project / "src" / "app.py"), str(project / "src" / "mod.py")])
    assert any("undefined_name" in e and "F821" in e for e in errors)


def test_generated_code_gets_no_api_keys(tmp_path, monkeypatch):
    # Anything the models wrote runs sandboxed; it must never see the
    # pipeline's own credentials (a debug dump would get committed and pushed).
    from tools.safety import run_safe_command, clean_env
    monkeypatch.setenv("FAKE_PROVIDER_API_KEY", "should-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "should-not-leak")
    probe = "python3 -c \"import os; print(sorted(k for k in os.environ if 'not-leak' in os.environ[k]))\""
    ok, out, _ = run_safe_command(probe, cwd=str(tmp_path), sandbox_dir=str(tmp_path))
    assert ok and out.strip() == "[]"
    # Unsandboxed pipeline commands (installs, git) keep the normal environment.
    ok, out, _ = run_safe_command(probe, cwd=str(tmp_path))
    assert ok and "FAKE_PROVIDER_API_KEY" in out
    assert "PATH" in clean_env() and "GITHUB_TOKEN" not in clean_env()


# ── static websites ──────────────────────────────────────────────────────────

def test_web_check_finds_unbuilt_css_and_missing_files(tmp_path):
    src = tmp_path / "site" / "src"
    (src / "css").mkdir(parents=True)
    (src / "css" / "styles.css").write_text("/* theme */\n@tailwind base;\n@tailwind utilities;\n")
    (src / "index.html").write_text(
        '<link rel="stylesheet" href="css/styles.css">\n<script src="js/main.js"></script>\n'
        '<a href="https://example.com">x</a> <a href="#contact">y</a> <a href="mailto:a@b.c">z</a>\n'
        '<link href="{{ url_for(\'static\', filename=\'x.css\') }}">\n')
    found = {(f["file"], f["line"], f["code"]) for f in sc.web_check(str(tmp_path / "site"))}
    assert found == {("src/css/styles.css", 2, "css-needs-build"), ("src/index.html", 2, "missing-file")}


def test_a_static_site_can_be_served(tmp_path):
    from tools.webapp import app_command
    (tmp_path / "index.html").write_text("<h1>hi</h1>")
    cmd = app_command(tmp_path, 8123)
    assert cmd[1:4] == ["-m", "http.server", "8123"]
    (tmp_path / "index.html").unlink()
    assert app_command(tmp_path, 8123) is None


def test_file_target_matches_where_files_are_written(tmp_path):
    from tools.file_ops import project_file_target, write_project_file
    for path in ("src/app.py", "app.py", "tests/test_a.py", "src/tests/test_b.py", "../tests/test_c.py",
                 "README.md", "src/css/site.css", "requirements.txt"):
        assert write_project_file(str(tmp_path), path, "x") == project_file_target(str(tmp_path), path), path
    assert project_file_target(str(tmp_path), "/etc/passwd") == ""


def test_update_tasks_cannot_wipe_earlier_sections():
    from agents.coder import _dropped
    page = "<main>\n" + "".join(f'<section id="s{i}">\n<p>{i}</p>\n</section>\n' for i in range(8)) + "</main>\n"
    only_contact = '<main>\n<section id="s7">\n<p>form</p>\n</section>\n</main>\n'
    assert _dropped(page, only_contact, "src/index.html").startswith("sections #s0, #s1")
    grown = page.replace("</main>", '<section id="s8"></section>\n</main>')
    assert _dropped(page, grown, "src/index.html") == ""
    code = "\n".join(f"line_{i} = {i}" for i in range(40))
    assert "of its 40 lines" in _dropped(code, "line_0 = 0\n", "src/app.py")
    assert _dropped("a = 1\n", "b = 2\n", "src/app.py") == ""   # small files are left alone


def test_an_unlinked_tailwind_file_is_not_reported(tmp_path):
    src = tmp_path / "site" / "src"
    src.mkdir(parents=True)
    (src / "input.css").write_text("@tailwind base;\n")        # leftover, nothing links it
    (src / "styles.css").write_text(":root { --accent: teal; }\n")
    (src / "index.html").write_text('<link rel="stylesheet" href="styles.css">\n')
    assert sc.web_check(str(tmp_path / "site")) == []


def test_update_tasks_cannot_delete_functions():
    from agents.coder import _dropped
    old = ("const siteContent = {};\nfunction renderServices(list) { return list; }\n"
           "const renderCases = (items) => items.map(String);\nfunction init() { renderServices([]); }\n")
    new = "function init() { console.log('ready'); }\nfunction animate() {}\n"
    assert _dropped(old, new, "src/js/main.js") == "functions renderCases, renderServices"
    assert _dropped(old, old + "function extra() {}\n", "src/js/main.js") == ""


def test_scripts_nothing_loads_are_reported(tmp_path):
    src = tmp_path / "site" / "src"
    (src / "js").mkdir(parents=True)
    (src / "js" / "data.js").write_text("const siteContent = {};\n")
    (src / "js" / "util.js").write_text("export const x = 1;\n")
    (src / "js" / "main.js").write_text("import { x } from './util.js';\n")
    (src / "index.html").write_text('<script type="module" src="js/main.js"></script>\n')
    found = [(f["file"], f["code"], f["severity"]) for f in sc.web_check(str(tmp_path / "site"))]
    assert found == [("src/js/data.js", "unused-script", "MEDIUM")]


def test_page_check_sees_empty_sections_and_errors(tmp_path):
    from tools import webapp
    if not webapp.browser():
        pytest.skip("no Chrome/Edge installed")
    src = tmp_path / "site" / "src"
    src.mkdir(parents=True)
    (src / "index.html").write_text(
        '<section id="about"><h2>About</h2><p>Real text about the person.</p></section>\n'
        '<section id="services"><h2>Services</h2><div id="grid"></div></section>\n'
        '<script src="missing.js"></script><script>undefinedFunction();</script>\n')
    found = [(f["code"], f["message"]) for f in sc.page_check(str(tmp_path / "site"))]
    assert ("empty-section", "section #services shows nothing but its heading once the page has loaded") in found
    assert any(code == "js-error" and "undefinedFunction" in msg for code, msg in found)
    assert any(code == "failed-load" and "404 /missing.js" in msg for code, msg in found)
    assert not any("#about" in msg for _, msg in found)


def test_page_check_reports_a_csp_that_blocks_the_page(tmp_path):
    # A security "fix" added a policy that blocked Tailwind's inline config and styles.
    from tools import webapp
    if not webapp.browser():
        pytest.skip("no Chrome/Edge installed")
    src = tmp_path / "site" / "src"
    src.mkdir(parents=True)
    (src / "index.html").write_text(
        '<head><meta http-equiv="Content-Security-Policy" content="default-src \'self\'"></head>\n'
        '<section id="about"><h2>About</h2><p>Real text about the person.</p></section>\n'
        '<script>document.title = "x";</script>\n')
    found = [(f["code"], f["message"]) for f in sc.page_check(str(tmp_path / "site"))]
    assert any(code == "csp-blocked" and "inline script" in msg for code, msg in found), found


def test_page_check_finds_text_nobody_can_read(tmp_path):
    # With its Tailwind config gone, a portfolio's "bg-accent text-white" buttons were white on white.
    from tools import webapp
    if not webapp.browser():
        pytest.skip("no Chrome/Edge installed")
    src = tmp_path / "site" / "src"
    src.mkdir(parents=True)
    (src / "index.html").write_text(
        '<section id="hero"><h1>Asha</h1><p>Digital marketing specialist in Bengaluru.</p>\n'
        '<a class="bg-accent" style="color:#fff">Contact me</a>\n'
        '<a style="color:#fff;background:#0d9488">View work</a>\n'
        '<div style="background-image:linear-gradient(#000,#333)"><span style="color:#fff">On a gradient</span></div>\n'
        '<span style="position:absolute;width:1px;height:1px;overflow:hidden;color:#fff">Skip link</span></section>\n')
    found = [f for f in sc.page_check(str(tmp_path / "site")) if f["code"] == "invisible-text"]
    assert len(found) == 1 and "“Contact me”" in found[0]["message"], found
    assert not any(t in found[0]["message"] for t in ("View work", "On a gradient", "Skip link", "Asha"))


def test_page_check_finds_headings_the_size_of_body_text(tmp_path):
    # Tailwind's CDN reset loads after the stylesheet: every heading came out 16px.
    from tools import webapp
    if not webapp.browser():
        pytest.skip("no Chrome/Edge installed")
    src = tmp_path / "site" / "src"
    src.mkdir(parents=True)
    page = ('<section id="hero"><h1>Asha</h1><p>Digital marketing in Bengaluru.</p></section>\n'
            '<section id="services"><h2>Services</h2><p>SEO, ads and analytics for small brands.</p></section>\n')
    (src / "index.html").write_text("<style>h1, h2 { font-size: inherit }</style>\n" + page)
    found = [f["message"] for f in sc.page_check(str(tmp_path / "site")) if f["code"] == "flat-headings"]
    assert found and "h1 “Asha” (16px)" in found[0], found
    (src / "index.html").write_text(page)   # the browser's own heading sizes are fine
    assert not [f for f in sc.page_check(str(tmp_path / "site")) if f["code"] == "flat-headings"]


def test_page_check_reads_the_page_in_dark_mode_too(tmp_path):
    # The text switched to white in dark mode; the section kept a fixed white background.
    from tools import webapp
    if not webapp.browser():
        pytest.skip("no Chrome/Edge installed")
    src = tmp_path / "site" / "src"
    src.mkdir(parents=True)
    (src / "index.html").write_text(
        '<style>:root{--text:#0f172a} @media (prefers-color-scheme: dark){:root{--text:#f1f5f9}}'
        'body{color:var(--text)} .surface{background:#fff}</style>\n'
        '<section id="about" class="surface"><h2>About Asha</h2><p>Digital marketing in Bengaluru.</p></section>\n')
    found = [f["message"] for f in sc.page_check(str(tmp_path / "site")) if f["code"] == "invisible-text"]
    # Named with the element painting the white behind it, so the fix goes there.
    assert len(found) == 1 and '“About Asha”, “Digital marketing in Bengaluru.” (dark mode, on ' \
                               '<section id="about" class="surface">)' in found[0], found
    assert found[0].count("“About Asha”") == 1   # fine in light mode


def test_a_big_page_is_never_shown_as_empty(tmp_path):
    # A 12.8 KB page over a 12,000-char view came out as "(no top-level
    # definitions)" — the reviewer called every section missing, twice.
    from agents.coder import _existing_code_context
    src = tmp_path / "src"
    (src / "js").mkdir(parents=True)
    (src / "css").mkdir()
    (src / "css" / "a.css").write_text("a { color: red; }\n")
    (src / "index.html").write_text('<section id="hero">' + "x" * 3000 + '</section><section id="about"></section>\n')
    (src / "js" / "main.js").write_text("function reveal() {}\n")
    view = _existing_code_context(str(tmp_path), budget=1000)
    assert "It is NOT empty — element ids: #hero, #about" in view
    assert "function reveal() {}" in view and "a { color: red; }" in view   # later files that fit are still shown


def test_merge_refuses_patches_that_lose_content(tmp_path):
    from agents.code_merge import run_code_merge
    src = tmp_path / "src"
    src.mkdir()
    (tmp_path / "docs").mkdir()
    page = "".join(f'<section id="s{i}"><h2>S{i}</h2><p>text</p></section>\n' for i in range(4))
    (src / "index.html").write_text(page)
    (src / "app.css").write_text("a { color: red; }\n")
    state = {"project_path": str(tmp_path), "agent_logs": [], "_security_findings": [], "_optimization_notes": [],
             "_security_fixed_files": {"src/index.html": "<main>\n<!-- Sections content remains identical -->\n</main>\n"},
             "_optimized_files": {"src/index.html": '<section id="s0">x</section>\n', "src/app.css": "a{color:red}\n"}}
    run_code_merge(state)
    assert (src / "index.html").read_text() == page           # both page rewrites refused
    assert (src / "app.css").read_text() == "a{color:red}\n"  # a real optimization still lands


def test_a_missing_photo_is_a_note_not_a_blocker(tmp_path):
    src = tmp_path / "site" / "src"
    src.mkdir(parents=True)
    (src / "index.html").write_text('<img src="assets/profile.jpg" alt="me">\n<script src="js/app.js"></script>\n')
    found = {(f["code"], f["severity"]) for f in sc.web_check(str(tmp_path / "site"))}
    assert found == {("missing-asset", "MEDIUM"), ("missing-file", "HIGH")}


def test_ellipsis_placeholders_are_never_written():
    from tools.file_blocks import parse_file_blocks
    lazy = ('=== FILE: src/index.html ===\n<main>\n<section id="hero">...</section>\n'
            '<section id="about">...</section>\n</main>\n=== END FILE ===\n'
            '=== FILE: src/js/main.js ===\nfunction a() {}\n...\n=== END FILE ===\n'
            '=== FILE: src/models.py ===\nclass P:\n    ...\n=== END FILE ===\n')
    assert list(parse_file_blocks(lazy)) == ["src/models.py"]   # Python's `...` is real code


def test_tailwind_classes_without_a_build_are_reported(tmp_path):
    src = tmp_path / "site" / "src"
    src.mkdir(parents=True)
    (src / "styles.css").write_text(".btn-primary { color: teal; }\n.card { padding: 1rem; }\n")
    utilities = "bg-gray-50 text-gray-900 max-w-[1100px] mx-auto px-4 py-8 rounded-lg shadow-md md:flex gap-4"
    (src / "index.html").write_text(f'<link rel="stylesheet" href="styles.css">\n<main class="{utilities}">'
                                    '<a class="btn-primary card">x</a></main>\n')
    found = [f for f in sc.web_check(str(tmp_path / "site")) if f["code"] == "tailwind-classes"]
    assert len(found) == 1 and found[0]["severity"] == "HIGH" and "btn-primary" not in found[0]["message"]
    # A page that styles itself with its own classes is fine.
    (src / "index.html").write_text('<link rel="stylesheet" href="styles.css">\n<a class="btn-primary card">x</a>\n')
    assert sc.web_check(str(tmp_path / "site")) == []


def test_tailwind_from_its_cdn_is_fine(tmp_path):
    src = tmp_path / "site" / "src"
    src.mkdir(parents=True)
    utilities = "bg-gray-50 text-gray-900 max-w-[1100px] mx-auto px-4 py-8 rounded-lg shadow-md md:flex gap-4"
    (src / "index.html").write_text('<head><script src="https://cdn.tailwindcss.com"></script></head>\n'
                                    f'<main class="{utilities}">x</main>\n')
    assert sc.web_check(str(tmp_path / "site")) == []


def test_framework_pages_nothing_builds_are_reported(tmp_path):
    src = tmp_path / "site" / "src"
    (src / "pages").mkdir(parents=True)
    (src / "pages" / "index.astro").write_text("---\n---\n<h1>Hi</h1>\n")
    found = [f for f in sc.web_check(str(tmp_path / "site")) if f["code"] == "needs-build"]
    assert len(found) == 1 and found[0]["severity"] == "HIGH" and "index.astro" in found[0]["message"]


def test_a_web_project_without_a_page_is_a_blocker(tmp_path):
    (tmp_path / "site" / "src").mkdir(parents=True)
    found = sc.page_check(str(tmp_path / "site"))
    assert [(f["code"], f["severity"]) for f in found] == [("no-entry-page", "HIGH")]
