"""Phase P1 tests: boilerplate removal, table preservation, facts-only filtering."""

from __future__ import annotations

from src.ingest import cleaner

HTML = """
<html>
  <head><title>HDFC Large Cap Fund - Direct - Growth</title>
    <style>.x{color:red}</style>
    <script>window.tracker = 1;</script>
  </head>
  <body>
    <nav>Home Funds About Us</nav>
    <aside class="cookie">We use cookies to improve your experience. Accept all cookies.</aside>
    <h1>Expense ratio</h1>
    <div>Expense ratio</div><div>1.03%</div>
    <div>Exit load</div><div>1% if redeemed within 1 year</div>
    <div>Benchmark</div><div>Nifty 100 TRI</div>
    <p>Very High risk.</p>
    <table id="returns">
      <tr><th>Over the past</th><th>Total investment</th><th>Would've become</th><th>Returns</th></tr>
      <tr><td>1 year</td><td>Rs 60,000</td><td>Rs 58,430</td><td>-2.62 %</td></tr>
    </table>
    <table id="holdings">
      <tr><th>Name</th><th>Sector</th><th>Instruments</th><th>Assets</th></tr>
      <tr><td>ICICI Bank Ltd</td><td>Financial</td><td>Equity</td><td>10.05%</td></tr>
    </table>
    <table id="peers">
      <tr><th>Name</th><th>1Y</th><th>3Y</th><th>Fund Size(Cr)</th></tr>
      <tr><td>Invesco India Large Cap Fund</td><td>+2.51%</td><td>+13.80%</td><td>2,020.76</td></tr>
    </table>
    <footer>About us | Privacy | Terms</footer>
  </body>
</html>
"""


def test_boilerplate_is_removed():
    result = cleaner.clean_html(HTML)
    text = result.text
    for junk in ("window.tracker", "color:red", "Home Funds About Us", "We use cookies", "Privacy | Terms"):
        assert junk not in text, junk
    assert result.title == "HDFC Large Cap Fund - Direct - Growth"


def test_factual_facts_survive():
    text = cleaner.clean_html(HTML).text
    for fact in ("Expense ratio", "1.03%", "1% if redeemed within 1 year", "Nifty 100 TRI", "Very High risk"):
        assert fact in text, fact


def test_performance_tables_are_dropped_and_marked():
    result = cleaner.clean_html(HTML)
    assert result.stats["tables_found"] == 3
    assert result.stats["tables_dropped"] == 2
    assert result.stats["tables_kept"] == 1
    assert cleaner.EXCLUSION_SENTINEL in result.text
    assert "-2.62" not in result.text
    assert "+2.51" not in result.text


def _table_block_containing(text: str, needle: str) -> str:
    """Return the `<TABLE n> ... </TABLE n>` block that contains `needle`."""
    for index in range(10):
        opener = cleaner.TABLE_OPEN.format(n=index)
        if opener not in text:
            continue
        block = text.split(opener, 1)[1].split(cleaner.TABLE_CLOSE.format(n=index), 1)[0]
        if needle in block:
            return block
    raise AssertionError(f"no table block containing {needle!r}")


def test_holdings_table_is_kept_with_sentinels():
    result = cleaner.clean_html(HTML)
    assert "ICICI Bank Ltd" in result.text
    assert "Financial" in result.text
    assert result.tables == [
        [
            ["Name", "Sector", "Instruments", "Assets"],
            ["ICICI Bank Ltd", "Financial", "Equity", "10.05%"],
        ]
    ]


def test_kept_table_repeats_the_header_row_in_text():
    result = cleaner.clean_html(HTML)
    block = _table_block_containing(result.text, "ICICI Bank Ltd")
    lines = block.strip().splitlines()
    assert lines[0] == "Name | Sector | Instruments | Assets"
    assert "ICICI Bank Ltd | Financial | Equity | 10.05%" in lines


def test_performance_prose_lines_are_dropped():
    html = "<body><p>Historic returns</p><p>Return calculator</p><p>Category average</p><p>Expense ratio 1.03%</p></body>"
    text = cleaner.clean_html(html).text
    assert "Historic returns" not in text
    assert "Return calculator" not in text
    assert "Category average" not in text
    assert "Expense ratio 1.03%" in text


def test_include_performance_keeps_everything():
    result = cleaner.clean_html(HTML, include_performance=True)
    assert result.stats["tables_dropped"] == 0
    assert result.stats["tables_kept"] == 3
    assert "-2.62" in result.text


def test_performance_table_detection_is_narrow():
    holdings = [["Name", "Sector", "Instruments", "Assets"], ["ICICI Bank Ltd", "Financial", "Equity", "10.05%"]]
    returns = [["Over the past", "Total investment", "Would've become"], ["1 year", "Rs 60,000", "Rs 58,430"]]
    assert cleaner.is_performance_table(holdings) is False
    assert cleaner.is_performance_table(returns) is True


def test_duplicate_consecutive_lines_are_collapsed():
    result = cleaner.clean_html("<body><div>1.03%</div><div>1.03%</div><div>1.04%</div></body>")
    assert result.text.splitlines() == ["1.03%", "1.04%"]


def test_normalize_ws_collapses_runs_and_newlines():
    assert cleaner.normalize_ws("  a \t b  \n\n\n c  ") == "a b\nc"


def test_pdf_text_rejoins_hyphenated_line_breaks():
    result = cleaner.clean_pdf_text("Expense\nratio is 1.03%\n\nExit load 1%\n")
    assert "Expenseratio" not in result.text
    assert "Expense" in result.text and "ratio is 1.03%" in result.text


def test_pdf_text_drops_performance_lines():
    result = cleaner.clean_pdf_text("Historic returns\nExpense ratio 1.03%\nRank 45\n")
    assert "Historic returns" not in result.text
    assert "Rank 45" not in result.text
    assert "Expense ratio 1.03%" in result.text


def test_return_ticker_fragments_are_dropped():
    html = (
        "<body><div>Returns on your systematic withdrawal plan</div>"
        "<div>+8.71</div><div>%</div><div>3Y annualised</div>"
        "<div>1D</div><div>1M</div><div>6M</div><div>1Y</div><div>3Y</div><div>5Y</div><div>All</div>"
        "<div>NAV: 25 Sep '26</div><div>&#8377;1,189.08</div>"
        "<div>Expense ratio</div><div>1.03%</div>"
        "<div>Min. for SIP</div><div>&#8377;100</div></body>"
    )
    result = cleaner.clean_html(html)
    text = result.text
    for gone in ("+8.71", "3Y annualised", "1D", "1M", "6M", "1Y", "3Y", "5Y", "All", "systematic withdrawal"):
        assert gone not in text.splitlines(), gone
    for kept in ("Expense ratio", "1.03%", "Min. for SIP", "100", "1,189.08"):
        assert kept in text, kept


def test_signed_figure_followed_by_bare_percent_is_dropped():
    # The live scheme pages separate the sign, the figure and the percent sign.
    kept, dropped = cleaner.finalise_lines(
        ["SWP calculator", "Credit", "Loan against securities", "+8.71", "%", "NAV: 25 Sep '26", "1,189.08"]
    )
    assert kept == ["SWP calculator", "Credit", "Loan against securities", "NAV: 25 Sep '26", "1,189.08"]
    assert dropped == 2


def test_signed_percentage_on_its_own_line_is_dropped():
    kept, _ = cleaner.finalise_lines(["Fund performance", "-1.42%"])
    assert kept == ["Fund performance"]


def test_negative_holding_weight_inside_a_table_row_is_kept():
    html = (
        "<body><table><tr><th>Name</th><th>Sector</th><th>Instruments</th><th>Assets</th></tr>"
        "<tr><td>Net Payables</td><td>Unspecified</td><td>Net Payables</td><td>-0.10%</td></tr>"
        "</table></body>"
    )
    result = cleaner.clean_html(html)
    assert "Net Payables | Unspecified | Net Payables | -0.10%" in result.text
    assert result.stats["tables_dropped"] == 0


def test_finalise_lines_counts_drops():
    kept, dropped = cleaner.finalise_lines(["Historic returns", "1.03%", "cookie banner", "Expense ratio"])
    assert kept == ["1.03%", "Expense ratio"]
    assert dropped == 2


def test_empty_html_does_not_raise():
    result = cleaner.clean_html("")
    assert result.text == ""
    assert result.stats["tables_found"] == 0
