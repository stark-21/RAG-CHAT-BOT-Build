"""Bounded conversation memory for retrieval scoping.

The window holds the last 10 prior questions. It is used for exactly one thing: deciding
which scheme a deictic follow-up belongs to, so "and its exit load?" is answered from the
fund the conversation is already about instead of whichever fund happens to rank highest
across all five.

Two things this deliberately does not do:

- **It does not concatenate the turns into the embedded query.** Averaging ten turns into
  one embedding blurs it toward the centroid of the conversation, and a question about exit
  load stops looking like a question about exit load. The current question is embedded on
  its own, exactly as before.
- **It does not reach generation.** The model still sees one question and its retrieved
  chunks, so conversation history never leaves the machine and a PII turn in the buffer
  cannot be sent to a hosted provider.

Scoping is therefore expressed through the existing `scheme_id` override on `retrieve()`
rather than as query rewriting. With no buffer passed, or with a self-contained query, the
retriever behaves identically to before.

Nothing is persisted. The buffer lives in `st.session_state` for the life of the session.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from src.ingest.sources import load_registry, resolve_scheme

MEMORY_WINDOW = 10


def expand_with_scheme(query: str, scheme_id: str | None) -> str:
    """Prefix the carried scheme's canonical name so the embedded query names the fund.

    A deictic follow-up embeds weakly on its own: "And its exit load?" scores 0.272 against
    the ELSS exit-load chunk, just under the 0.30 gate, because "and" and "its" carry no
    meaning. Naming the fund fixes it - the same question scores 0.851 - and it works
    because the chunks are titled "<scheme name> - <section>", so the scheme name is
    literally what the vector is keyed on.

    This adds one scheme name, not the window. Ten turns concatenated would blur the
    embedding toward the centre of the conversation; one name sharpens it.

    Returns the query unchanged when there is no carried scheme, when the query already
    names it, or when the registry cannot supply a name - a lookup failure must never be
    the reason a question goes unanswered.
    """
    if not scheme_id:
        return query
    try:
        spec = load_registry().scheme(scheme_id)
    except Exception:
        return query
    name = (getattr(spec, "scheme_name", "") or "").strip()
    if not name or name.lower() in query.lower():
        return query
    return f"{name} {query}"


@dataclass(frozen=True)
class Turn:
    """One prior question and the scheme it resolved to, if any."""

    query: str
    scheme_id: str | None


class TurnBuffer:
    """The last `window` questions, used to scope follow-ups to a scheme.

    Only user questions are recorded. An assistant turn carries no new referent, so
    storing it would let a scheme stick around for longer than the user meant it to.
    """

    def __init__(self, window: int = MEMORY_WINDOW) -> None:
        self._window = max(0, int(window))
        self._turns: deque[Turn] = deque(maxlen=self._window or 1)

    def __len__(self) -> int:
        return len(self._turns)

    @property
    def window(self) -> int:
        return self._window

    def remember(self, query: str, scheme_id: str | None = None) -> None:
        """Record a question and the scheme it resolved to.

        `scheme_id` is resolved here when not supplied, so a caller that has not run the
        retriever yet still records a usable referent.
        """
        text = (query or "").strip()
        if not text or self._window == 0:
            return
        if scheme_id is None:
            scheme_id = resolve_scheme(text)
        self._turns.append(Turn(query=text, scheme_id=scheme_id))

    def carried_scheme_id(self, query: str) -> str | None:
        """The scheme a follow-up inherits, or None when the query stands on its own.

        A question that names a scheme is self-contained, so nothing is carried forward:
        forcing an earlier scheme onto it would override what the user just asked. Only a
        question that names no scheme of its own inherits the most recent one, which is
        what makes "and its exit load?" resolve after "tell me about the ELSS fund".
        """
        if resolve_scheme(query or ""):
            return None
        for turn in reversed(self._turns):
            if turn.scheme_id:
                return turn.scheme_id
        return None

    def turns(self) -> tuple[Turn, ...]:
        """Oldest first, for tests and for showing the user what is being remembered."""
        return tuple(self._turns)

    def clear(self) -> None:
        self._turns.clear()
