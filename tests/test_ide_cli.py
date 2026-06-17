from __future__ import annotations

from latticememory.ide.lattice_ops import list_verticals, proxy_doctor


def test_list_verticals_includes_prompt_firewall():
    rows = list_verticals()

    assert any(row["class"] == "LatticePromptFirewall" for row in rows)
    assert any(row["class"] == "LatticeTrainingCleaner" for row in rows)


def test_proxy_doctor_reports_unreachable():
    result = proxy_doctor(host="127.0.0.1", port=9)

    assert result["reachable"] is False
    assert "error" in result
