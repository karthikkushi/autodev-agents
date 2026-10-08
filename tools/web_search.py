import concurrent.futures
try:
    # The package was renamed to `ddgs`; the old `duckduckgo_search` now
    # returns nothing (or junk), which left every research step working blind.
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    from duckduckgo_search import DDGS
from rich.console import Console

console = Console()


SEARCH_TIMEOUT = 20
# Module-level pool with a hard limit: a hung search froze the tester for
# minutes. The abandoned request is left to finish (or die) on its own.
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="web-search")


def _search(query: str, max_results: int) -> list:
    return list(DDGS(timeout=10).text(query, max_results=max_results) or [])


def search_web(query: str, max_results: int = 5) -> str:
    console.print(f"[yellow]🔍 Searching: {query}[/yellow]")
    try:
        rows = _POOL.submit(_search, query, max_results).result(timeout=SEARCH_TIMEOUT)
    except concurrent.futures.TimeoutError:
        return f"Search failed: no answer within {SEARCH_TIMEOUT}s"
    except Exception as e:
        return f"Search failed: {e}"
    results = [f"**{r.get('title', '')}**\n{r.get('body', '')}\nURL: {r.get('href', '')}\n" for r in rows]
    return "\n---\n".join(results) if results else "No results found."


def search_github(query: str) -> str:
    return search_web(f"site:github.com {query}", max_results=3)


def search_docs(library: str, topic: str) -> str:
    return search_web(f"{library} {topic} documentation example 2026", max_results=4)
