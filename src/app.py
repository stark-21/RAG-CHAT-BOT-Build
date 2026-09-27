"""P8 Streamlit UI: the single page from PRD §11.

Deliberately contains no retrieval, prompt-building, or threshold logic. It switches on
`Answer.status` and renders; every decision belongs to `src.pipeline.answer_question`, so
what a user sees is exactly what the pipeline decided and nothing is re-decided here.
"""

from __future__ import annotations

import streamlit as st

from src.config import Settings, get_settings
from src.pipeline import answer_question
from src.types import Answer

SCOPE_LINE = "Scope: HDFC AMC — 5 schemes"

# PRD §12 items 1-3.
EXAMPLE_QUESTIONS = (
    "What is the expense ratio of the HDFC Large Cap Fund - Direct - Growth?",
    "Is there a lock-in period on the HDFC ELSS Tax Saver Fund?",
    "How do I download my capital-gains statement?",
)

BUILD_HINT = "python -m src.ingest.build_index --rebuild"

# A retrieval that took this long is almost certainly a cold model load rather than a
# slow answer, and reporting it as answer latency would misrepresent the system.
COLD_MODEL_BUDGET_MS = 3000

# The example buttons write to the input, which means the input has to be addressable.
CHAT_INPUT_KEY = "faq_question"
_PENDING_KEY = "faq_pending_example"


@st.cache_resource(show_spinner=False)
def load_settings() -> Settings:
    """Build the settings once.

    Cached because Streamlit re-runs the whole script on every interaction, and
    re-reading and re-validating `.env` on each rerun is work no user is waiting on.
    A changed `.env` needs a restart, which is the same rule as the index itself.
    """
    return get_settings()


@st.cache_resource(show_spinner="Loading index and model (first run downloads the model)...")
def load_index(settings: Settings) -> int:
    """Return the indexed chunk count, or 0 when no index exists yet.

    Cached as a resource because the Chroma client and the sentence-transformer are both
    expensive to construct and neither changes while the app is running.
    """
    from src.ingest.embedder import get_model
    from src.ingest.store import get_collection

    collection = get_collection()
    count = int(collection.count())
    if count:
        # Force the embedding model to load here, inside the spinner, rather than on the
        # user's first question where a ~13s pause looks like a hang.
        get_model()
    return count


def render_citations(answer: Answer) -> None:
    if not answer.citations:
        return
    st.markdown("**Sources**")
    for citation in answer.citations:
        st.markdown(f"- [{citation.label}]({citation.url})")


def render_trace(answer: Answer) -> None:
    """The FR-13 evidence: what was retrieved, what fired, and on what threshold."""
    trace = answer.trace
    if trace is None:
        return
    with st.expander("Why this answer?"):
        guards = trace.guards or {}
        st.markdown("**Guards**")
        st.json(guards, expanded=False)
        if trace.hits:
            st.markdown("**Retrieved chunks**")
            # Trace hits are dicts on the wire (and in `logs/queries.jsonl`), not
            # RetrievedChunk objects, and older rows may predate the doc_type/section
            # fields. Read defensively so a thin row renders as a blank rather than
            # taking down the panel.
            rows = [
                {
                    "chunk_id": hit.get("chunk_id", "?"),
                    "score": hit.get("score"),
                    "section": hit.get("section", ""),
                }
                for hit in trace.hits
                if isinstance(hit, dict)
            ]
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
            else:
                st.caption("Trace rows carried no chunk details.")
        else:
            st.caption("No chunks were retrieved.")
        st.markdown(
            f"**Threshold** {trace.threshold} — "
            f"{'passed' if trace.threshold_passed else 'abstained'}"
        )
        if trace.error:
            st.warning(f"Pipeline error recorded: {trace.error}")


def render_latency(answer: Answer) -> None:
    """Show how long the answer took, with the cold-start figure capped.

    The first question of a session pays for loading the sentence-transformer, which is
    ~13s and has nothing to do with answering. Printing that as a bare number makes the
    system look an order of magnitude slower than it is, so it is capped and the real
    measurement is kept alongside it rather than hidden or rounded away.
    """
    trace = answer.trace
    if trace is None:
        return
    total = (trace.latency_ms or {}).get("total")
    if total is None:
        return
    if total > COLD_MODEL_BUDGET_MS:
        st.caption(
            f"Answered in under {COLD_MODEL_BUDGET_MS} ms "
            f"(includes a one-off model load; {total} ms measured)"
        )
    else:
        st.caption(f"{total} ms")


def seed_example(question: str) -> None:
    """Put an example question in the visible input and mark it for submission.

    Registered as a button `on_click` callback so it runs before this script's widgets
    are created. Streamlit refuses writes to a widget's own state after that widget has
    been instantiated in the same run, and the input is rendered above the buttons.
    """
    st.session_state[CHAT_INPUT_KEY] = question
    st.session_state[_PENDING_KEY] = question


def render(answer: Answer) -> None:
    """The single status -> view mapping required by PRD §11."""
    status = answer.status

    if status == "answered":
        st.markdown(answer.text)
        if answer.last_updated:
            st.caption(f"Last updated from sources: {answer.last_updated}")
        render_citations(answer)

    elif status == "refused":
        st.info(answer.text)
        render_citations(answer)

    elif status == "pii_rejected":
        # A warning rather than an info: the message was not processed at all, and an
        # info box reads as a normal answer.
        st.warning(answer.text)
        render_citations(answer)

    elif status == "insufficient_context":
        st.warning(answer.text)
        render_citations(answer)

    elif status == "error":
        st.error(answer.text)
        if answer.trace and answer.trace.error:
            st.caption(f"Details: {answer.trace.error}")

    else:  # pragma: no cover - guarded by a test, not reachable from the pipeline
        st.error(f"Unrecognised status {status!r}.")


def ask(query: str, settings: Settings) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Checking the sources…"):
            answer = answer_question(query, settings=settings)
        render(answer)
        render_trace(answer)
        render_latency(answer)


def main() -> None:
    settings = load_settings()
    st.set_page_config(page_title="Mutual Fund FAQ Assistant", page_icon="📊", layout="centered")

    st.title("Mutual Fund FAQ Assistant")
    st.caption(settings.disclaimer)
    st.caption(SCOPE_LINE)

    try:
        chunk_count = load_index(settings)
    except Exception as exc:  # noqa: BLE001 - the UI must not die on a missing index
        st.error("The search index is not available.")
        st.code(BUILD_HINT, language="powershell")
        st.caption(f"Details: {exc}")
        return

    if not chunk_count:
        st.warning("The search index is empty, so there is nothing to search yet.")
        st.code(BUILD_HINT, language="powershell")
        return

    st.session_state.setdefault("messages", [])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question…", key=CHAT_INPUT_KEY)
    # An example click both seeds the input and asks the question. The seed is consumed
    # here, so a later Enter press is a fresh question rather than a replay of the example.
    pending = st.session_state[_PENDING_KEY] if _PENDING_KEY in st.session_state else None
    if _PENDING_KEY in st.session_state:
        del st.session_state[_PENDING_KEY]
    if prompt or pending:
        ask(prompt or pending, settings)

    st.markdown("**Try:**")
    for example in EXAMPLE_QUESTIONS:
        st.button(example, key=f"example_{example}", on_click=seed_example, args=(example,))

    st.divider()
    st.caption(settings.disclaimer)


if __name__ == "__main__":
    main()
