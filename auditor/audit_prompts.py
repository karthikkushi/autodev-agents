"""Per-agent critique prompts for the Shadow Auditor.

Each prompt asks the local model to find issues, missing cases, and
improvements in that agent's output — nothing more. Keep responses short;
these become phone notifications, not reports.
"""

AUDIT_PROMPTS = {
    "planner": """You are a silent auditor reviewing an architecture document written by another AI.
Look for: missing error paths, no auth/security consideration, scalability gaps, unclear module boundaries.
If everything looks solid, reply with just the word NONE.
Otherwise reply with 1-3 short bullet points (max 200 words total), most important first.

ARCHITECTURE DOCUMENT:
{output}""",

    "coder": """You are a silent auditor reviewing source code written by another AI.
Look for: missing input validation, SQL injection / XSS risks, no exception handling, hardcoded secrets, missing edge cases.
If everything looks solid, reply with just the word NONE.
Otherwise reply with 1-3 short bullet points (max 200 words total), most important first.

SOURCE CODE:
{output}""",

    "reviewer": """You are a silent auditor reviewing a code review report written by another AI.
Look for: issues marked low-severity that actually look critical, missing security findings, incomplete test coverage notes.
If everything looks solid, reply with just the word NONE.
Otherwise reply with 1-3 short bullet points (max 200 words total), most important first.

REVIEW REPORT:
{output}""",

    "tester": """You are a silent auditor reviewing test results written by another AI.
Look for: tests passing but coverage is shallow, no edge case tests, missing integration tests.
If everything looks solid, reply with just the word NONE.
Otherwise reply with 1-3 short bullet points (max 200 words total), most important first.

TEST RESULTS:
{output}""",

    "final_review": """You are a silent auditor reviewing a project's final README / report written by another AI.
Look for: README missing key setup steps, no API docs, incomplete deployment instructions.
If everything looks solid, reply with just the word NONE.
Otherwise reply with 1-3 short bullet points (max 200 words total), most important first.

PROJECT REPORT:
{output}""",

    "security": """You are a silent auditor reviewing a security scan report written by another AI.
Look for: findings that were downgraded to low/medium severity but look critical, obvious
vulnerability classes the scan didn't mention (auth bypass, SSRF, insecure deserialization).
If everything looks solid, reply with just the word NONE.
Otherwise reply with 1-3 short bullet points (max 200 words total), most important first.

SECURITY SCAN REPORT:
{output}""",

    "optimizer": """You are a silent auditor reviewing an optimization pass written by another AI.
Look for: changes that might have altered behavior (not just performance), premature
optimization that hurts readability, missed obvious hotspots.
If everything looks solid, reply with just the word NONE.
Otherwise reply with 1-3 short bullet points (max 200 words total), most important first.

OPTIMIZATION NOTES:
{output}""",

    "test_writer": """You are a silent auditor reviewing a generated test suite written by another AI.
Look for: tests that only cover the happy path, no negative/edge-case tests, tests that
assert too loosely to catch real regressions.
If everything looks solid, reply with just the word NONE.
Otherwise reply with 1-3 short bullet points (max 200 words total), most important first.

TEST SUITE SUMMARY:
{output}""",

    "doc_writer": """You are a silent auditor reviewing a generated README written by another AI.
Look for: missing setup steps, install commands that won't actually work, unexplained
prerequisites, missing "how to run" instructions.
If everything looks solid, reply with just the word NONE.
Otherwise reply with 1-3 short bullet points (max 200 words total), most important first.

README:
{output}""",
}

# Agent -> relative path(s), within a project folder, of the output to audit.
# "coder" is special-cased (reads all files under src/) in shadow_auditor.py.
AUDIT_TARGETS = {
    "planner": "docs/03_architecture.md",
    "security": "docs/05b_security.md",
    "reviewer": "docs/07_review.md",
    "optimizer": "docs/06b_optimizations.md",
    "test_writer": "docs/07b_tests_written.md",
    "tester": "docs/08_test_results.md",
    "doc_writer": "README.md",
    "final_review": "docs/10_final_report.md",
}
