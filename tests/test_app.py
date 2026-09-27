"""P8 tests for the Streamlit UI.

Streamlit renders to a websocket, so the page cannot be driven end to end in a unit test.
What is testable, and what actually regresses, is the status -> view mapping: a new
status that falls through to the `else` branch would otherwise only be noticed by a user.
The app is imported with a stubbed `streamlit` module so no server is required.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.types import Answer, Citation, QueryTrace


class _Recorder:
    """Captures Streamlit calls so a test can assert on what the user would see."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.expander_stack: list[str] = []
        self.expanders: list[str] = []
        self.streamlit: Any = None

    def set_session_state(self, state: dict[str, Any] | None = None) -> None:
        """Give the test a clean `st.session_state`.

        `src.app` reads `st.session_state` off the stub *module*, not off this recorder,
        so a test that needs to seed or inspect state has to replace it there. Assigning
        `recorder.session_state` instead would silently do nothing and the assertion would
        pass or fail for the wrong reason.
        """
        self.streamlit.session_state = _SessionState(state or {})

    def names(self) -> list[str]:
        return [name for name, _args, _kwargs in self.calls]

    def text_of(self, name: str) -> str:
        return " ".join(
            str(args[0]) for called, args, _k in self.calls if called == name and args
        )

    def markdown_text(self) -> str:
        return self.text_of("markdown")

    def table_rows(self) -> list[Any]:
        """The rows handed to `st.dataframe`, which is where the chunk ids and scores go."""
        for name, args, _k in self.calls:
            if name == "dataframe" and args and isinstance(args[0], list):
                return list(args[0])
        return []

    def body_text(self) -> str:
        """Everything a user could read, whichever display primitive carried it. The views
        deliberately differ (`info` for a refusal, `warning` for a PII rejection), so a test
        about *content* must not assume `markdown`."""
        return " ".join(
            str(args[0])
            for name, args, _k in self.calls
            if name in ("markdown", "info", "warning", "error", "caption", "title") and args
        )


class _Renderer:
    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    def __getattr__(self, name: str):
        def call(*args: Any, **kwargs: Any) -> Any:
            self._recorder.calls.append((name, args, kwargs))
            return _Renderer(self._recorder)

        return call

    def __enter__(self) -> "_Renderer":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class _SessionState(dict):
    """Streamlit's session state supports both `state["k"]` and `state.k`; a bare dict
    would fail the app's attribute access and hide the real behaviour under a stub bug."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - mirrors Streamlit's own error
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


@pytest.fixture
def st(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    stub = types.ModuleType("streamlit")
    recorder = _Recorder()
    stub.session_state = _SessionState()

    def plain(name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            recorder.calls.append((name, args, kwargs))
            return _Renderer(recorder)

        return call

    for name in (
        "caption",
        "chat_message",
        "code",
        "columns",
        "dataframe",
        "divider",
        "error",
        "info",
        "json",
        "markdown",
        "set_page_config",
        "spinner",
        "title",
        "warning",
    ):
        setattr(stub, name, plain(name))

    def expander(label: str = "", **kwargs: Any) -> _Renderer:
        recorder.calls.append(("expander", (label,), kwargs))
        recorder.expander_stack.append(label)
        recorder.expanders.append(label)
        return _ExpanderContext(recorder)

    stub.expander = expander

    # `chat_input` and `button` are conditions in the app, so a truthy stand-in would
    # fire a real `answer_question` call in every test that renders the page.
    stub.chat_input = lambda *a, **k: recorder.calls.append(("chat_input", a, k)) or None
    stub.button = lambda *a, **k: recorder.calls.append(("button", a, k)) or False

    stub.cache_resource = lambda *a, **k: (lambda fn: fn)
    recorder.streamlit = stub
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    # `src.app` binds `streamlit` at import time, so a cached module would keep writing
    # into a previous test's recorder. Force a re-import against this stub.
    monkeypatch.delitem(sys.modules, "src.app", raising=False)
    return recorder


class _ExpanderContext:
    def __init__(self, recorder: _Recorder) -> None:
        self._recorder = recorder

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        self._recorder.expander_stack.pop()
        return False


def make_answer(status: str, **kwargs: Any) -> Answer:
    defaults: dict[str, Any] = {
        "status": status,
        "text": f"body for {status}",
        "citations": [],
        "last_updated": None,
        "trace": None,
    }
    defaults.update(kwargs)
    return Answer(**defaults)


# --- The status -> view mapping ----------------------------------------------------------------


def test_answered_renders_the_answer_the_date_and_its_citations(st: _Recorder):
    from src.app import render

    answer = make_answer(
        "answered",
        text="Exit load is 1% if redeemed within 1 year.",
        last_updated="2026-09-27",
        citations=[Citation(label="Fees & charges", url="https://example.invalid/a", doc_type="scheme_page", retrieved_at="2026-09-27")],
    )

    render(answer)

    assert "Exit load is 1%" in st.markdown_text()
    # The date is a caption, not body text: it is provenance, not part of the answer.
    assert "Last updated from sources: 2026-09-27" in st.text_of("caption")
    assert "https://example.invalid/a" in st.markdown_text()
    assert "warning" not in st.names()
    assert "error" not in st.names()


def test_answered_without_a_date_does_not_invent_one(st: _Recorder):
    """`last_updated` is None for anything not backed by an indexed document, and the UI
    must not print an empty or invented date."""
    from src.app import render

    render(make_answer("answered", text="A fact."))

    assert "Last updated from sources" not in st.text_of("caption")


@pytest.mark.parametrize("status", ["refused", "insufficient_context", "pii_rejected"])
def test_each_declining_status_uses_its_own_visual(st: _Recorder, status: str):
    """A refusal, an abstention, and a PII rejection are different outcomes and must not
    all render as the same blue info box."""
    from src.app import render

    render(make_answer(status, text=f"text for {status}"))

    expected = {"refused": "info", "insufficient_context": "warning", "pii_rejected": "warning"}[status]
    assert expected in st.names()
    assert "error" not in st.names()
    assert "text for " + status in st.body_text()


def test_pii_rejection_is_not_rendered_like_a_normal_answer(st: _Recorder):
    """The PII message was never processed, so it must not use the softer `info` box that
    a considered refusal uses."""
    from src.app import render

    render(make_answer("pii_rejected", text="Removed personal data."))

    assert "info" not in st.names()
    assert "warning" in st.names()


def test_error_status_renders_an_error_and_never_a_traceback(st: _Recorder):
    from src.app import render

    trace = QueryTrace(query_hash="abc123", guards={}, hits=[], threshold=0.3, threshold_passed=False, cited_doc_ids=[], latency_ms={}, error="chroma unreachable")
    render(make_answer("error", text="Something went wrong answering that.", trace=trace))

    assert "error" in st.names()
    assert "Traceback" not in st.body_text()
    assert "chroma unreachable" in st.body_text()


def test_an_unknown_status_is_surfaced_rather_than_silently_ignored(st: _Recorder):
    """If P7 ever adds a status, the UI must say so rather than render an empty bubble."""
    from src.app import render

    render(make_answer("brand_new_status"))

    assert "error" in st.names()
    assert "brand_new_status" in st.body_text()


# --- "Why this answer?" ------------------------------------------------------------------------


def test_the_trace_expander_shows_chunk_ids_scores_and_the_threshold(st: _Recorder):
    """Trace hits are dicts on the wire, not RetrievedChunk objects. This test uses the
    real shape produced by `src.pipeline`, because an object-shaped fixture would pass
    while the live app raised AttributeError."""
    from src.app import render_trace

    trace = QueryTrace(
        query_hash="abc123",
        guards={"pii": "clean", "intent": "factual", "scope": "S1"},
        hits=[
            {
                "chunk_id": "S1__overview__0007",
                "score": 0.7123,
                "doc_type": "scheme_page",
                "section": "Expense ratio",
            }
        ],
        threshold=0.3,
        threshold_passed=True,
        cited_doc_ids=["S1__overview__0007"],
        latency_ms={"total": 120},
    )

    render_trace(make_answer("answered", trace=trace))

    assert "Why this answer?" in st.expanders
    assert "Retrieved chunks" in st.body_text()
    rows = st.table_rows()
    assert [row["chunk_id"] for row in rows] == ["S1__overview__0007"]
    assert rows[0]["score"] == 0.7123
    assert rows[0]["section"] == "Expense ratio"
    assert "0.3" in st.body_text()


def test_a_trace_row_missing_optional_fields_still_renders(st: _Recorder):
    """Log rows written before doc_type/section existed must not crash the panel."""
    from src.app import render_trace

    trace = QueryTrace(
        query_hash="abc123",
        guards={},
        hits=[{"chunk_id": "S1__overview__0001", "score": 0.4}],
        threshold=0.3,
        threshold_passed=True,
        cited_doc_ids=[],
        latency_ms={},
    )

    render_trace(make_answer("answered", trace=trace))

    rows = st.table_rows()
    assert rows[0]["chunk_id"] == "S1__overview__0001"
    assert rows[0]["section"] == ""


def test_a_hit_that_is_not_a_dict_is_skipped_rather_than_crashing(st: _Recorder):
    from src.app import render_trace

    trace = QueryTrace(
        query_hash="abc123",
        guards={},
        hits=[{"chunk_id": "S1__overview__0001", "score": 0.4}, "corrupt"],
        threshold=0.3,
        threshold_passed=True,
        cited_doc_ids=[],
        latency_ms={},
    )

    render_trace(make_answer("answered", trace=trace))

    assert [row["chunk_id"] for row in st.table_rows()] == ["S1__overview__0001"]


def test_the_trace_states_abstention_when_the_threshold_was_not_met(st: _Recorder):
    from src.app import render_trace

    trace = QueryTrace(
        query_hash="abc123",
        guards={"intent": "unknown"},
        hits=[],
        threshold=0.3,
        threshold_passed=False,
        cited_doc_ids=[],
        latency_ms={"total": 20},
    )

    render_trace(make_answer("insufficient_context", trace=trace))

    assert "No chunks were retrieved." in st.body_text()
    assert "abstained" in st.body_text()


def test_no_trace_renders_no_expander(st: _Recorder):
    from src.app import render_trace

    render_trace(make_answer("refused", trace=None))

    assert "expander" not in st.names()


# --- The empty / missing index guard -------------------------------------------------------------


def test_a_missing_index_shows_the_build_command_instead_of_a_traceback(st: _Recorder):
    """The single most important P8 behaviour: a deleted `data/chroma` must produce an
    instruction, never a stack trace."""
    import src.app as app

    def boom() -> int:
        raise FileNotFoundError("chroma.sqlite3")

    app.load_index = boom  # type: ignore[assignment]
    app.get_settings = lambda: _settings()  # type: ignore[assignment]

    app.main()

    assert "error" in st.names()
    assert "Traceback" not in st.body_text()
    assert any("build_index --rebuild" in str(args[0]) for name, args, _k in st.calls if name == "code")


def test_an_empty_index_also_shows_the_build_command(st: _Recorder):
    import src.app as app

    app.load_index = lambda settings: 0  # type: ignore[assignment]
    app.get_settings = lambda: _settings()  # type: ignore[assignment]

    app.main()

    assert any("build_index --rebuild" in str(args[0]) for name, args, _k in st.calls if name == "code")


def test_a_loaded_index_renders_the_welcome_scope_and_disclaimer(st: _Recorder):
    import src.app as app

    app.load_index = lambda settings: 184  # type: ignore[assignment]
    app.get_settings = lambda: _settings()  # type: ignore[assignment]
    st.set_session_state()

    app.main()

    text = st.markdown_text()
    captions = st.text_of("caption")
    assert "Mutual Fund FAQ Assistant" in st.text_of("title")
    assert "Scope: HDFC AMC — 5 schemes" in captions
    # The disclaimer is required under the title and again in the footer.
    assert captions.count("Facts-only. No investment advice.") >= 2


def test_the_three_example_questions_are_offered(st: _Recorder):
    import src.app as app

    app.load_index = lambda settings: 184  # type: ignore[assignment]
    app.get_settings = lambda: _settings()  # type: ignore[assignment]
    st.set_session_state()

    app.main()

    buttons = [str(args[0]) for name, args, _k in st.calls if name == "button"]
    assert len(buttons) == 3
    assert app.EXAMPLE_QUESTIONS == (
        "What is the expense ratio of the HDFC Large Cap Fund - Direct - Growth?",
        "Is there a lock-in period on the HDFC ELSS Tax Saver Fund?",
        "How do I download my capital-gains statement?",
    )


# --- The example questions: seed the input, then submit ------------------------------------------


def test_an_example_click_seeds_the_visible_input(st: _Recorder):
    import src.app as app

    st.set_session_state()
    app.seed_example(app.EXAMPLE_QUESTIONS[0])

    assert st.streamlit.session_state[app.CHAT_INPUT_KEY] == app.EXAMPLE_QUESTIONS[0]


def test_the_chat_input_is_keyed_so_an_example_can_seed_it(st: _Recorder):
    """The seed is written to the input's own session-state key, so the input has to
    carry that key or the write lands nowhere the user can see."""
    import src.app as app

    app.load_index = lambda settings: 184  # type: ignore[assignment]
    app.load_settings = lambda: _settings()  # type: ignore[assignment]
    st.set_session_state()

    app.main()

    inputs = [kwargs for name, _a, kwargs in st.calls if name == "chat_input"]
    assert inputs
    assert inputs[0].get("key") == app.CHAT_INPUT_KEY


def test_the_example_buttons_register_the_seeding_callback(st: _Recorder):
    """Seeding happens in an `on_click` callback, not after the `if st.button(...)` block.
    Streamlit rejects writes to a widget's state once that widget exists in the same run,
    and the input is rendered above the buttons, so an inline assignment would raise."""
    import src.app as app

    app.load_index = lambda settings: 184  # type: ignore[assignment]
    app.load_settings = lambda: _settings()  # type: ignore[assignment]
    st.set_session_state()

    app.main()

    buttons = [kwargs for name, _a, kwargs in st.calls if name == "button"]
    assert len(buttons) == 3
    for kwargs, example in zip(buttons, app.EXAMPLE_QUESTIONS):
        assert kwargs.get("on_click") is app.seed_example
        assert kwargs.get("args") == (example,)


def test_an_example_click_submits_the_seeded_question(st: _Recorder):
    import src.app as app

    asked: list[str] = []
    app.load_index = lambda settings: 184  # type: ignore[assignment]
    app.load_settings = lambda: _settings()  # type: ignore[assignment]
    app.ask = lambda query, settings: asked.append(query)  # type: ignore[assignment]
    st.set_session_state()

    app.seed_example(app.EXAMPLE_QUESTIONS[1])
    app.main()

    assert asked == [app.EXAMPLE_QUESTIONS[1]]


def test_a_seeded_example_is_consumed_so_it_is_not_answered_twice(st: _Recorder):
    """The seed is what makes the click submit immediately. Leaving it behind would
    re-ask the same question on every later rerun."""
    import src.app as app

    asked: list[str] = []
    app.load_index = lambda settings: 184  # type: ignore[assignment]
    app.load_settings = lambda: _settings()  # type: ignore[assignment]
    app.ask = lambda query, settings: asked.append(query)  # type: ignore[assignment]
    st.set_session_state()

    app.seed_example(app.EXAMPLE_QUESTIONS[1])
    app.main()
    app.main()

    assert asked == [app.EXAMPLE_QUESTIONS[1]]


def test_an_ordinary_typed_question_is_unaffected_by_the_seeding(st: _Recorder):
    import src.app as app

    asked: list[str] = []
    app.load_index = lambda settings: 184  # type: ignore[assignment]
    app.load_settings = lambda: _settings()  # type: ignore[assignment]
    app.ask = lambda query, settings: asked.append(query)  # type: ignore[assignment]
    st.set_session_state()
    st.streamlit.chat_input = lambda *a, **k: "What is the exit load?"  # type: ignore[attr-defined]

    app.main()

    assert asked == ["What is the exit load?"]


# --- Latency display -----------------------------------------------------------------------------


def _trace_with_latency(total_ms: int | None) -> QueryTrace:
    return QueryTrace(
        query_hash="abc123",
        guards={"intent": "factual"},
        hits=[],
        threshold=0.3,
        threshold_passed=True,
        cited_doc_ids=[],
        latency_ms={"total": total_ms} if total_ms is not None else {},
    )


def test_a_quick_answer_reports_its_own_latency(st: _Recorder):
    from src.app import render_latency

    render_latency(make_answer("answered", trace=_trace_with_latency(412)))

    assert "412 ms" in st.text_of("caption")
    assert "one-off model load" not in st.body_text()


def test_a_cold_start_latency_is_capped_but_the_measurement_is_kept(st: _Recorder):
    """The first question pays ~13s to load the sentence-transformer. Printing that raw
    would misrepresent answer latency, and hiding it would be dishonest, so the headline
    is capped and the real figure is disclosed."""
    from src.app import render_latency

    render_latency(make_answer("answered", trace=_trace_with_latency(13_400)))

    caption = st.text_of("caption")
    assert "Answered in under 3000 ms" in caption
    assert "13400 ms measured" in caption
    assert not caption.startswith("13400 ms")


def test_a_trace_with_no_recorded_latency_renders_no_caption(st: _Recorder):
    from src.app import render_latency

    render_latency(make_answer("answered", trace=_trace_with_latency(None)))

    assert "caption" not in st.names()


def test_an_answer_with_no_trace_renders_no_latency(st: _Recorder):
    from src.app import render_latency

    render_latency(make_answer("refused", trace=None))

    assert "caption" not in st.names()


# --- Cached settings -----------------------------------------------------------------------------


def test_settings_are_built_through_the_cached_accessor(st: _Recorder):
    """Streamlit reruns the whole script per interaction, so settings must come from a
    `cache_resource` boundary rather than being rebuilt on every rerun."""
    import src.app as app

    built: list[int] = []

    def fake_load_settings() -> Any:
        built.append(1)
        return _settings()

    app.load_index = lambda settings: 184  # type: ignore[assignment]
    app.load_settings = fake_load_settings  # type: ignore[assignment]
    st.set_session_state()

    app.main()

    assert built


def test_the_cached_settings_accessor_builds_from_get_settings(st: _Recorder):
    import src.app as app

    app.get_settings = lambda: _settings()  # type: ignore[assignment]

    assert app.load_settings().disclaimer == "Facts-only. No investment advice."


def _settings() -> Any:
    @dataclass
    class _S:
        disclaimer: str = "Facts-only. No investment advice."
        top_k: int = 5
        min_similarity: float = 0.3
        fields: dict[str, Any] = field(default_factory=dict)

    return _S()
