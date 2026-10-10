"""Doc pages that must never appear in training data.

The 20 questions in tests/questions.json were written from these pages. If they
were in training, the evaluation would measure memorization, not learning.

Matching is by file name (without .mdx) across every raw doc folder, so the
same topic in the Pages Router docs is also excluded.
"""

from pathlib import Path

EVAL_PAGE_STEMS = {
    "dynamic-routes",
    "layout",
    "page",
    "loading",
    "error",
    "not-found",
    "route",
    "proxy",
    "route-groups",
    "parallel-routes",
    "use-client",
    "use-server",
    "use-pathname",
    "use-search-params",
    "redirect",
    "permanentRedirect",
    "revalidatePath",
    "generate-metadata",
    "generate-static-params",
    "image",
    "link",
    "04-glossary",
}


def is_eval_source(path):
    """True if this doc page is one of the evaluation source pages."""
    return Path(path).stem in EVAL_PAGE_STEMS
