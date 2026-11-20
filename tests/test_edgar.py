import pytest

from whymoved.data.edgar import EdgarClient, cause_label, clean_filing_html, item_prefix


def test_cause_priority():
    assert cause_label(["2.02", "9.01"]) == "earnings"
    assert cause_label(["5.02", "2.02"]) == "earnings"  # earnings outranks leadership
    assert cause_label(["1.01", "9.01"]) == "mna"
    assert cause_label(["5.02"]) == "leadership"
    assert cause_label(["2.03"]) == "financing"
    assert cause_label(["2.06"]) == "restructuring_impairment"
    assert cause_label(["5.07"]) == "governance"
    assert cause_label(["9.01"]) == "other" and cause_label(["7.01", "9.01"]) == "other"
    assert cause_label(["1.03"]) == "other" and cause_label([]) == "other"


def test_clean_html_cuts_cover_page():
    html = ("<html><style>x{}</style><body><p>UNITED STATES SEC FORM 8-K cover</p>"
            "<p>Item 2.02  Results of Operations</p><script>bad()</script><p>Revenue up.</p></body></html>")
    t = clean_filing_html(html)
    assert t.startswith("Item 2.02") and "bad" not in t and "cover" not in t and "Revenue" in t
    assert clean_filing_html("<p>no items here</p>") == "no items here"
    assert len(clean_filing_html("<p>Item 1.01 " + "a" * 5000 + "</p>")) == 3000


def test_item_prefix():
    assert item_prefix(["2.02", "9.01", "9.99"]).startswith("2.02 Results of Operations")


def test_requires_user_agent(monkeypatch):
    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError, match="EDGAR_USER_AGENT"):
        EdgarClient()
