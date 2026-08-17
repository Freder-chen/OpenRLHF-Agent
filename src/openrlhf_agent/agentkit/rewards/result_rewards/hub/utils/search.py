"""Text helpers for Search-R1 style exact-match QA scoring."""

from __future__ import annotations

import re
import string
from typing import Optional

# Synchronize with search-r1
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_FINAL_ANSWER_RE = re.compile(r"(?im)^\s*Answer:\s*(.+?)\s*$")
_PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    text = text.lower().translate(_PUNCT_TRANSLATION)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def extract_final_answer(response: str) -> Optional[str]:
    matches = _FINAL_ANSWER_RE.findall(response)
    if matches:
        answer = matches[-1].strip()
        return answer or None
    return None
