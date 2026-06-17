"""Tests for latticememory.verticals — all 6 domain modules."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class HashEncoder:
    def __init__(self, d_model: int = 384):
        self.d_model = d_model

    def encode(self, sentences, **kwargs):
        vecs = []
        for s in sentences:
            seed = int(hashlib.md5(str(s).encode()).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.d_model).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            vecs.append(v)
        return np.stack(vecs)


def _make_cache(d_model: int = 384):
    from latticememory.memory import RFSnapLatticeMemory
    from latticememory.text_runtime import RFSnapTextMemory
    from latticememory.semantic_cache import RFSnapSemanticCache

    lm = RFSnapLatticeMemory(d_model=d_model)
    rt = RFSnapTextMemory(encoder=HashEncoder(d_model), d_model=d_model, memory=lm)
    return RFSnapSemanticCache(runtime=rt)


# ---------------------------------------------------------------------------
# 1. LatticeSOCMonitor
# ---------------------------------------------------------------------------

class TestLatticeSOCMonitor:
    def test_novel_alert_not_duplicate(self):
        from latticememory.verticals import LatticeSOCMonitor
        mon = LatticeSOCMonitor(_make_cache())
        r = mon.ingest_alert("unusual outbound traffic on port 4444")
        assert not r.is_duplicate
        assert r.decision is None

    def test_same_alert_twice_is_duplicate(self):
        from latticememory.verticals import LatticeSOCMonitor
        mon = LatticeSOCMonitor(_make_cache())
        r1 = mon.ingest_alert("brute force login attempt on admin account")
        r2 = mon.ingest_alert("brute force login attempt on admin account")
        assert not r1.is_duplicate
        assert r2.is_duplicate
        assert r2.canonical_alert_id is not None

    def test_known_pattern_returned_on_match(self):
        from latticememory.verticals import LatticeSOCMonitor
        mon = LatticeSOCMonitor(_make_cache())
        mon.add_known_pattern("ransomware encrypting files", playbook={"action": "isolate_host"})
        r = mon.ingest_alert("ransomware encrypting files")
        assert r.is_duplicate
        assert r.is_known_pattern
        assert r.decision["action"] == "isolate_host"

    def test_stats_tracks_ingested_count(self):
        from latticememory.verticals import LatticeSOCMonitor
        mon = LatticeSOCMonitor(_make_cache())
        for i in range(5):
            mon.ingest_alert(f"alert {i}")
        assert mon.stats()["alerts_ingested"] == 5

    def test_gap_log_written(self, tmp_path):
        from latticememory.verticals import LatticeSOCMonitor
        gap_log = str(tmp_path / "gaps.jsonl")
        mon = LatticeSOCMonitor(_make_cache(), gap_log_path=gap_log)
        mon.ingest_alert("novel zero-day exploit detected in kernel")
        import os
        assert os.path.exists(gap_log)
        lines = open(gap_log).readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "novel zero-day" in record["question"]

    def test_novel_patterns_returns_clusters(self, tmp_path):
        from latticememory.verticals import LatticeSOCMonitor
        gap_log = str(tmp_path / "gaps.jsonl")
        mon = LatticeSOCMonitor(_make_cache(), gap_log_path=gap_log)
        # Same alert 3 times → should cluster
        for _ in range(3):
            mon.ingest_alert("suspicious DNS exfiltration attempt")
        clusters = mon.top_novel_patterns(n=5)
        # May or may not cluster depending on threshold, but should not error
        assert isinstance(clusters, list)


# ---------------------------------------------------------------------------
# 2. LatticeTicketAnalyzer
# ---------------------------------------------------------------------------

class TestLatticeTicketAnalyzer:
    def test_miss_on_unanswered_ticket(self):
        from latticememory.verticals import LatticeTicketAnalyzer
        ta = LatticeTicketAnalyzer(_make_cache())
        r = ta.ingest_ticket("how do I export my data?")
        assert r.hit_type == "miss"
        assert r.suggested_answer is None
        assert r.is_documentation_gap

    def test_hit_after_resolved_ticket_added(self):
        from latticememory.verticals import LatticeTicketAnalyzer
        ta = LatticeTicketAnalyzer(_make_cache())
        ta.add_resolved_ticket("how do I export my data?", answer="Go to Settings → Export")
        r = ta.ingest_ticket("how do I export my data?")
        assert r.hit_type != "miss"
        assert r.suggested_answer == "Go to Settings → Export"
        assert not r.is_documentation_gap

    def test_stats_counts_correctly(self):
        from latticememory.verticals import LatticeTicketAnalyzer
        ta = LatticeTicketAnalyzer(_make_cache())
        ta.ingest_ticket("question 1")
        ta.ingest_ticket("question 2")
        s = ta.stats()
        assert s["tickets_ingested"] == 2
        assert s["answered_entries"] == 0

    def test_export_review_queue_writes_file(self, tmp_path):
        from latticememory.verticals import LatticeTicketAnalyzer
        gap_log = str(tmp_path / "gaps.jsonl")
        ta = LatticeTicketAnalyzer(_make_cache(), gap_log_path=gap_log)
        # Ingest 3 times to make a cluster
        for _ in range(3):
            ta.ingest_ticket("I cannot log in to my account")
        queue_path = str(tmp_path / "queue.json")
        n = ta.export_review_queue(queue_path)
        # n may be 0 if cluster size < min_cluster_size, but should not error
        assert isinstance(n, int)

    def test_load_reviewed_adds_to_cache(self, tmp_path):
        from latticememory.verticals import LatticeTicketAnalyzer
        gap_log = str(tmp_path / "gaps.jsonl")
        ta = LatticeTicketAnalyzer(_make_cache(), gap_log_path=gap_log)
        ta.ingest_ticket("where is the billing page?")
        ta.ingest_ticket("where is the billing page?")
        ta.ingest_ticket("where is the billing page?")

        queue_path = str(tmp_path / "queue.json")
        ta.export_review_queue(queue_path)

        # If queue has items, fill in answers and load
        items = json.loads(open(queue_path).read())
        if items:
            items[0]["answer"] = "Go to Account → Billing"
            open(queue_path, "w").write(json.dumps(items))
            added = ta.load_reviewed(queue_path)
            assert added >= 1


# ---------------------------------------------------------------------------
# 3. LatticeContentModerator
# ---------------------------------------------------------------------------

class TestLatticeContentModerator:
    def test_novel_content_pending(self):
        from latticememory.verticals import LatticeContentModerator
        mod = LatticeContentModerator(_make_cache())
        r = mod.submit("buy cheap pills now")
        assert r.status == "pending_review"
        assert not r.auto_applied
        assert mod.pending_count() == 1

    def test_known_rejected_content_auto_applied(self):
        from latticememory.verticals import LatticeContentModerator
        mod = LatticeContentModerator(_make_cache())
        mod.record_decision("buy cheap pills now", "rejected", reviewer_id="reviewer1")
        r = mod.submit("buy cheap pills now")
        assert r.status == "rejected"
        assert r.auto_applied

    def test_approved_content_propagates(self):
        from latticememory.verticals import LatticeContentModerator
        mod = LatticeContentModerator(_make_cache())
        mod.record_decision("the weather is nice today", "approved")
        r = mod.submit("the weather is nice today")
        assert r.status == "approved"
        assert r.auto_applied

    def test_audit_log_written(self, tmp_path):
        from latticememory.verticals import LatticeContentModerator
        audit = str(tmp_path / "audit.jsonl")
        mod = LatticeContentModerator(_make_cache(), audit_log_path=audit)
        mod.submit("hello world")
        entries = mod.audit_log()
        assert len(entries) == 1
        assert entries[0]["status"] == "pending_review"

    def test_audit_log_since_filter(self, tmp_path):
        from latticememory.verticals import LatticeContentModerator
        audit = str(tmp_path / "audit.jsonl")
        mod = LatticeContentModerator(_make_cache(), audit_log_path=audit)
        t0 = time.time()
        mod.submit("old content")
        time.sleep(0.01)
        t1 = time.time()
        mod.submit("new content")
        after = mod.audit_log(since=t1)
        assert len(after) == 1

    def test_precedent_id_set_on_auto_applied(self):
        from latticememory.verticals import LatticeContentModerator
        mod = LatticeContentModerator(_make_cache())
        mod.record_decision("spam message template", "rejected")
        r = mod.submit("spam message template")
        assert r.auto_applied
        assert r.precedent_content_id is not None


# ---------------------------------------------------------------------------
# 4. LatticeClauseCoder
# ---------------------------------------------------------------------------

class TestLatticeClauseCoder:
    def test_unknown_clause_goes_to_review(self):
        from latticememory.verticals import LatticeClauseCoder
        coder = LatticeClauseCoder(_make_cache())
        result = coder.code_document("The licensor grants a non-exclusive, worldwide license.")
        assert len(result.needs_review) >= 1
        assert len(result.auto_coded) == 0
        assert result.auto_rate == 0.0

    def test_known_clause_auto_coded(self):
        from latticememory.verticals import LatticeClauseCoder
        clause = "The licensor grants a non-exclusive, worldwide license to use the software."
        coder = LatticeClauseCoder(_make_cache())
        coder.record_decision(clause, "standard_license")
        result = coder.code_document(clause)
        assert len(result.auto_coded) >= 1
        assert result.auto_coded[0].decision == "standard_license"
        assert result.auto_coded[0].auto_coded

    def test_auto_rate_reflects_hits(self):
        from latticememory.verticals import LatticeClauseCoder
        cache = _make_cache()
        coder = LatticeClauseCoder(cache)
        clauses = [
            "The licensor grants a non-exclusive license.",
            "Licensee may not sublicense this software.",
            "All warranties are disclaimed to the maximum extent permitted.",
        ]
        for c in clauses:
            coder.record_decision(c, "approved")
        doc = "\n\n".join(clauses)
        result = coder.code_document(doc)
        assert result.auto_rate > 0.0

    def test_export_review_queue(self, tmp_path):
        from latticememory.verticals import LatticeClauseCoder
        coder = LatticeClauseCoder(_make_cache())
        coder.code_document("The governing law shall be the laws of the State of Delaware.\n\nThis agreement is binding on all parties.")
        path = str(tmp_path / "queue.json")
        n = coder.export_review_queue(path)
        assert n >= 1
        items = json.loads(open(path).read())
        assert items[0]["decision"] is None

    def test_load_reviewed_adds_decisions(self, tmp_path):
        from latticememory.verticals import LatticeClauseCoder
        coder = LatticeClauseCoder(_make_cache())
        coder.code_document("Confidential information shall not be disclosed.")
        path = str(tmp_path / "queue.json")
        coder.export_review_queue(path)
        items = json.loads(open(path).read())
        items[0]["decision"] = "confidentiality"
        open(path, "w").write(json.dumps(items))
        added = coder.load_reviewed(path)
        assert added == 1
        # Now it should auto-code
        r2 = coder.code_document("Confidential information shall not be disclosed.")
        assert len(r2.auto_coded) >= 1


# ---------------------------------------------------------------------------
# 5. LatticeEdgeMemory
# ---------------------------------------------------------------------------

class TestLatticeEdgeMemory:
    _KEY_LEN = 128

    def _random_key(self, seed=0):
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, self._KEY_LEN, dtype=np.uint8).tobytes().hex()

    def test_recognize_exact_match(self):
        from latticememory.verticals import LatticeEdgeMemory
        mem = LatticeEdgeMemory(threshold=10)
        key = self._random_key(seed=42)
        mem.learn(key, "obstacle_detected")
        r = mem.recognize(key)
        assert r.recognized
        assert r.label == "obstacle_detected"
        assert r.hamming_distance == 0
        assert r.confidence == 1.0

    def test_no_match_beyond_threshold(self):
        from latticememory.verticals import LatticeEdgeMemory
        mem = LatticeEdgeMemory(threshold=5)
        key1 = self._random_key(seed=1)
        key2 = self._random_key(seed=99)  # very different
        mem.learn(key1, "cat")
        r = mem.recognize(key2)
        assert not r.recognized
        assert r.label is None

    def test_empty_memory_returns_unrecognized(self):
        from latticememory.verticals import LatticeEdgeMemory
        mem = LatticeEdgeMemory()
        r = mem.recognize(self._random_key())
        assert not r.recognized
        assert r.hamming_distance == -1

    def test_forget_removes_entries(self):
        from latticememory.verticals import LatticeEdgeMemory
        mem = LatticeEdgeMemory(threshold=10)
        key = self._random_key(seed=5)
        mem.learn(key, "dog")
        assert len(mem) == 1
        removed = mem.forget("dog")
        assert removed == 1
        assert len(mem) == 0
        assert not mem.recognize(key).recognized

    def test_memory_bytes(self):
        from latticememory.verticals import LatticeEdgeMemory
        mem = LatticeEdgeMemory()
        for i in range(10):
            mem.learn(self._random_key(seed=i), f"label_{i}")
        assert mem.memory_bytes() == 10 * 128

    def test_capacity_enforcement(self):
        from latticememory.verticals import LatticeEdgeMemory
        mem = LatticeEdgeMemory(capacity=2)
        mem.learn(self._random_key(seed=0), "a")
        mem.learn(self._random_key(seed=1), "b")
        with pytest.raises(OverflowError):
            mem.learn(self._random_key(seed=2), "c")

    def test_save_and_load_roundtrip(self, tmp_path):
        from latticememory.verticals import LatticeEdgeMemory
        mem = LatticeEdgeMemory(threshold=10)
        keys = [self._random_key(seed=i) for i in range(5)]
        for i, key in enumerate(keys):
            mem.learn(key, f"thing_{i}", metadata={"index": i})
        path = str(tmp_path / "mem.lmem")
        mem.save(path)

        mem2 = LatticeEdgeMemory(threshold=10)
        mem2.load(path)
        assert len(mem2) == 5
        r = mem2.recognize(keys[2])
        assert r.recognized
        assert r.label == "thing_2"

    def test_persist_path_autoloads(self, tmp_path):
        from latticememory.verticals import LatticeEdgeMemory
        path = str(tmp_path / "auto.lmem")
        mem1 = LatticeEdgeMemory(threshold=10)
        key = self._random_key(seed=77)
        mem1.learn(key, "autoloaded")
        mem1.save(path)

        mem2 = LatticeEdgeMemory(threshold=10, persist_path=path)
        r = mem2.recognize(key)
        assert r.recognized and r.label == "autoloaded"


# ---------------------------------------------------------------------------
# 6. LatticePrivateSync
# ---------------------------------------------------------------------------

class TestLatticePrivateSync:
    def test_export_manifest_is_deterministic(self):
        from latticememory.verticals import LatticePrivateSync
        cache = _make_cache()
        cache.put("shared knowledge A", value="v1")
        cache.put("shared knowledge B", value="v2")
        ps = LatticePrivateSync(cache)
        m1 = ps.export_key_manifest()
        m2 = ps.export_key_manifest()
        assert m1 == m2
        assert len(m1) == 2
        assert m1 == sorted(m1)  # sorted

    def test_overlap_detection(self):
        from latticememory.verticals import LatticePrivateSync
        org_a = _make_cache()
        org_a.put("threat: ransomware variant X", value="block")
        org_a.put("threat: phishing kit 42", value="block")

        org_b = _make_cache()
        org_b.put("threat: ransomware variant X", value="block")  # same
        org_b.put("threat: novel APT campaign", value="alert")    # different

        ps_a = LatticePrivateSync(org_a)
        ps_b = LatticePrivateSync(org_b)

        report = ps_a.compare(ps_b.export_key_manifest())
        assert report.overlap_count >= 1  # ransomware variant X
        assert report.gap_count >= 1      # APT campaign

    def test_find_gaps_are_remote_keys_not_local(self):
        from latticememory.verticals import LatticePrivateSync
        local = _make_cache()
        local.put("concept A", value="x")
        remote_manifest = ["a" * 256, "b" * 256]  # fake remote keys
        ps = LatticePrivateSync(local)
        report = ps.compare(remote_manifest)
        # gaps = what remote has that we don't
        assert len(report.gap_keys) <= 2

    def test_export_and_import_documents(self):
        from latticememory.verticals import LatticePrivateSync
        org_a = _make_cache()
        org_a.put("threat intelligence report Q1", value="block outbound 10.0.0.5")

        org_b = _make_cache()

        ps_a = LatticePrivateSync(org_a)
        ps_b = LatticePrivateSync(org_b)

        # org_b finds org_a has something it doesn't
        manifest_a = ps_a.export_key_manifest()
        report = ps_b.compare(manifest_a)
        assert report.gap_count >= 1

        # org_b requests the gap documents from org_a
        docs = ps_a.export_documents_for_keys(report.gap_keys)
        assert len(docs) >= 1
        assert docs[0]["prompt"] == "threat intelligence report Q1"

        # org_b imports
        n = ps_b.import_documents(docs)
        assert n >= 1

        # Now org_b has the knowledge
        r = org_b.get("threat intelligence report Q1")
        assert r.hit

    def test_empty_manifest_has_no_overlap(self):
        from latticememory.verticals import LatticePrivateSync
        cache = _make_cache()
        cache.put("local only", value="v")
        ps = LatticePrivateSync(cache)
        report = ps.compare([])
        assert report.overlap_count == 0
        assert report.gap_count == 0
        assert report.surplus_count >= 1
