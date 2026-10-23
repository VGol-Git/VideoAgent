"""Meaningful regression tests for denominator and evidence-scoring failure modes."""

import copy
import math
import unittest

from clipscout.evaluation import bootstrap_ci, evaluate, retrieval_eval, temporal_iou, token_f1


def question(identifier="q1", *, answerable=True, intervals=None, category="localization", **overrides):
    record = {
        "id": identifier, "question": f"Question {identifier}?", "category": category,
        "split": "test", "answerable": answerable,
        "reference_answers": ["red toolbox"] if answerable else [],
        "evidence_intervals": ([[10, 20]] if intervals is None else intervals) if answerable else [],
    }
    return {**record, **overrides}


def prediction(identifier="q1", *, intervals=None, **overrides):
    record = {
        "question_id": identifier, "answer": "red toolbox", "status": "answered",
        "citations": [{"start": start, "end": end, "evidence_id": f"e{index}"} for index, (start, end) in enumerate([[10, 20]] if intervals is None else intervals)],
        "tool_calls": 2, "frames_used": 4, "latency_seconds": 0.2, "trace": [],
    }
    return {**record, **overrides}


def score(questions, predictions, **kwargs):
    return evaluate(questions, predictions, bootstrap_samples=50, **kwargs)


class LexicalMetricsTests(unittest.TestCase):
    def test_unicode_punctuation_case_and_repeated_tokens(self):
        self.assertEqual(token_f1("КРАСНЫЙ, ящик!", "красный ящик"), 1.0)
        self.assertAlmostEqual(token_f1("red red toolbox", "red toolbox"), 0.8)
        self.assertEqual(token_f1("", "toolbox"), 0.0)
        self.assertEqual(token_f1("", ""), 1.0)

    def test_any_reference_and_required_term_alternatives(self):
        q = question(reference_answers=["red toolbox", "a crimson chest"], required_terms=[["red", "crimson"], ["toolbox", "chest"]])
        result = score([q], [prediction(answer="A crimson chest.")])
        self.assertEqual(result["summary"]["exact_match"], 1.0)
        result = score([q], [prediction(answer="The chest was crimson.")])
        self.assertEqual(result["summary"]["lexical_correctness_proxy"], 1.0)
        self.assertEqual(result["summary"]["exact_match"], 0.0)

    def test_required_terms_do_not_match_substrings(self):
        q = question(required_terms=[["red"]])
        result = score([q], [prediction(answer="one hundred boxes")])
        self.assertEqual(result["summary"]["lexical_correctness_proxy"], 0.0)


class TemporalMetricsTests(unittest.TestCase):
    def test_iou_identity_disjoint_and_partial(self):
        self.assertEqual(temporal_iou([1, 2], [1, 2]), 1.0)
        self.assertEqual(temporal_iou([1, 2], [2, 3]), 0.0)
        self.assertAlmostEqual(temporal_iou([0, 10], [5, 15]), 1 / 3)
        self.assertEqual(temporal_iou([0, 1e308], [0, 1e308]), 1.0)

    def test_long_span_does_not_receive_full_grounding_credit(self):
        report = score([question()], [prediction(intervals=[[0, 100]])])
        self.assertEqual(report["summary"]["temporal_iou"], 0.1)
        self.assertEqual(report["summary"]["temporal_hit_at_0_3"], 0.0)

    def test_single_citation_cannot_satisfy_two_hops(self):
        q = question(intervals=[[0, 10], [10, 20]], category="temporal")
        report = score([q], [prediction(intervals=[[0, 20]])])
        self.assertEqual(report["summary"]["temporal_iou"], 0.25)
        self.assertEqual(report["summary"]["temporal_hit_at_0_5"], 1.0)
        self.assertEqual(report["summary"]["multi_hop_recall_at_0_5"], 0.5)
        self.assertEqual(report["summary"]["multi_hop_success_at_0_5"], 0.0)

    def test_duplicate_geometry_cannot_satisfy_two_hops(self):
        q = question(intervals=[[0, 10], [10, 20]])
        p = prediction(intervals=[[0, 20], [0, 20]])
        report = score([q], [p])
        self.assertEqual(report["per_question"][0]["unique_citation_interval_count"], 1)
        self.assertEqual(report["summary"]["multi_hop_success_at_0_5"], 0.0)

    def test_both_hops_and_irrelevant_citation_precision(self):
        q = question(intervals=[[0, 10], [20, 30]])
        report = score([q], [prediction(intervals=[[20, 30], [0, 10], [60, 70]])])
        self.assertEqual(report["summary"]["temporal_iou"], 1.0)
        self.assertEqual(report["summary"]["multi_hop_success_at_0_5"], 1.0)
        self.assertAlmostEqual(report["summary"]["citation_iou_precision"], 2 / 3)

    def test_matching_uses_global_assignment_instead_of_greedy(self):
        # First reference locally prefers [0, 10], but the second needs it.
        q = question(intervals=[[0, 10], [0, 6]])
        report = score([q], [prediction(intervals=[[0, 10], [6, 10]])])
        self.assertAlmostEqual(report["summary"]["temporal_iou"], 0.5)
        self.assertEqual(report["summary"]["multi_hop_success_at_0_3"], 1.0)

    def test_onset_error_conditional_denominator(self):
        qs = [question("a"), question("b"), question("c")]
        ps = [prediction("a", intervals=[[12, 20]]), prediction("b", intervals=[[40, 50]])]
        summary = score(qs, ps)["summary"]
        self.assertEqual(summary["onset_error_seconds"], 2.0)
        self.assertEqual(summary["onset_error_example_count"], 1)
        self.assertEqual(summary["grounding_example_count"], 3)
        self.assertAlmostEqual(summary["temporal_iou"], 0.8 / 3)

    def test_abstention_does_not_gain_grounding_credit_from_attached_citations(self):
        result = score([question()], [prediction(status="insufficient_evidence")])
        self.assertEqual(result["summary"]["temporal_iou"], 0.0)
        self.assertIsNone(result["summary"]["onset_error_seconds"])


class DenominatorAndBudgetTests(unittest.TestCase):
    def test_missing_and_error_predictions_stay_in_denominator(self):
        qs = [question("a"), question("b"), question("c")]
        ps = [prediction("a"), prediction("b", status="error")]
        summary = score(qs, ps)["summary"]
        self.assertAlmostEqual(summary["token_f1"], 1 / 3)
        self.assertAlmostEqual(summary["exact_match"], 1 / 3)
        self.assertAlmostEqual(summary["answerable_coverage"], 1 / 3)
        self.assertEqual(summary["missing_prediction_count"], 1)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["denominators"]["answer_metrics"], 3)
        self.assertEqual(summary["confidence_intervals"]["token_f1"]["n"], 3)

    def test_unanswerable_abstention_error_and_selective_risk(self):
        qs = [question("a"), question("b", answerable=False, category="unanswerable"), question("c", answerable=False, category="unanswerable"), question("d", answerable=False, category="unanswerable")]
        ps = [prediction("a"), prediction("b", status="insufficient_evidence", intervals=[]), prediction("c", status="error", intervals=[]), prediction("d", intervals=[])]
        summary = score(qs, ps)["summary"]
        self.assertAlmostEqual(summary["unanswerable_abstention_accuracy"], 1 / 3)
        self.assertEqual(summary["task_success_proxy"], 0.5)
        self.assertEqual(summary["selective_risk_proxy"], 0.5)
        self.assertEqual(summary["selective_risk_denominator"], 2)

    def test_no_answer_means_undefined_selective_risk(self):
        summary = score([question()], [prediction(status="insufficient_evidence")])["summary"]
        self.assertIsNone(summary["selective_risk_proxy"])
        self.assertEqual(summary["selective_risk_denominator"], 0)

    def test_budget_boundary_and_failed_trace_calls_count(self):
        calls = [{"tool": "inspect", "args": {}, "ok": False, "result": "failure"}] * 5
        report = score([question("a"), question("b")], [prediction("a", tool_calls=4, frames_used=16), prediction("b", tool_calls=1, frames_used=17, trace=calls)])
        summary = report["summary"]
        self.assertEqual(summary["tool_budget_violation_count"], 1)
        self.assertEqual(summary["frame_budget_violation_count"], 1)
        self.assertEqual(summary["tool_call_underreport_count"], 1)
        self.assertEqual(summary["total_tool_calls"], 9)
        self.assertEqual(summary["budget_violation_rate"], 0.5)

    def test_categories_retain_failed_rows_and_inputs_are_not_mutated(self):
        qs = [question("a"), question("b", category="visual")]
        ps = [prediction("a")]
        before = copy.deepcopy((qs, ps))
        report = score(qs, ps)
        self.assertEqual(report["per_category"]["visual"]["exact_match"], 0.0)
        self.assertEqual((qs, ps), before)

    def test_empty_input_has_defined_counts_and_undefined_means(self):
        summary = score([], [])["summary"]
        self.assertEqual(summary["example_count"], 0)
        self.assertIsNone(summary["token_f1"])
        self.assertIsNone(summary["confidence_intervals"]["token_f1"]["low"])


class InputValidationTests(unittest.TestCase):
    def test_duplicate_question_prediction_and_unknown_prediction_ids(self):
        for qs, ps in [([question(), question()], []), ([question()], [prediction(), prediction()]), ([question()], [prediction("unknown")])]:
            with self.subTest(qs=qs, ps=ps), self.assertRaises(ValueError):
                score(qs, ps)

    def test_invalid_reference_or_citation_intervals(self):
        for span in ([1, 1], [2, 1], [-1, 2], [0, math.nan], [0, math.inf], [True, 2]):
            with self.subTest(span=span), self.assertRaises(ValueError):
                score([question(intervals=[span])], [])
            with self.subTest(citation=span), self.assertRaises(ValueError):
                score([question()], [prediction(intervals=[span])])

    def test_same_source_can_support_distinct_subintervals(self):
        p = prediction(intervals=[[0, 10], [10, 20]])
        p["citations"][1]["evidence_id"] = p["citations"][0]["evidence_id"]
        report = score([question(intervals=[[0, 10], [10, 20]])], [p])
        self.assertEqual(report["summary"]["multi_hop_success_at_0_5"], 1.0)

    def test_invalid_usage_and_limits(self):
        for field, value in [("tool_calls", -1), ("frames_used", 1.5), ("latency_seconds", math.nan), ("latency_seconds", -0.1)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                score([question()], [prediction(**{field: value})])
        with self.assertRaises(ValueError):
            score([question()], [], max_tool_calls=-1)


class BootstrapAndRetrievalTests(unittest.TestCase):
    def test_bootstrap_deterministic_and_constant_case(self):
        first = bootstrap_ci([0.0, 0.5, 1.0], samples=100, seed=7)
        self.assertEqual(first, bootstrap_ci([0.0, 0.5, 1.0], samples=100, seed=7))
        self.assertEqual(first["mean"], 0.5)
        self.assertLessEqual(first["low"], first["mean"])
        self.assertGreaterEqual(first["high"], first["mean"])
        constant = bootstrap_ci([1.0], samples=10)
        self.assertEqual((constant["low"], constant["high"]), (1.0, 1.0))

    def test_bootstrap_invalid_inputs_and_empty(self):
        self.assertEqual(bootstrap_ci([])["n"], 0)
        for options in ({"samples": 0}, {"confidence": 1.0}):
            with self.assertRaises(ValueError):
                bootstrap_ci([1.0], **options)
        with self.assertRaises(ValueError):
            bootstrap_ci([math.nan])

    def test_retrieval_queries_original_questions_without_reference_leakage(self):
        calls = []
        def search(text, *, k):
            calls.append((text, k))
            return [{"start": 10, "end": 20}, {"start": 50, "end": 60}]
        qs = [question("a"), question("b", answerable=False, category="unanswerable")]
        result = retrieval_eval(qs, search, k=1)
        self.assertEqual(calls, [("Question a?", 1), ("Question b?", 1)])
        self.assertEqual(result["summary"]["hit_at_1_iou_0_5"], 1.0)
        self.assertEqual(result["summary"]["grounding_example_count"], 1)
        self.assertEqual(result["per_question"][0]["unique_citation_interval_count"], 1)

    def test_retrieval_failures_stay_in_denominator(self):
        def search(text, *, k):
            if "b?" in text:
                raise RuntimeError("backend unavailable")
            return {"hits": [{"start": 10, "end": 20}]}
        result = retrieval_eval([question("a"), question("b")], search)
        self.assertEqual(result["summary"]["hit_at_5_iou_0_5"], 0.5)
        self.assertEqual(result["summary"]["retrieval_error_count"], 1)
        self.assertIn("backend unavailable", result["per_question"][1]["retrieval_error"])

    def test_retrieval_multihop_and_invalid_hit(self):
        q = question(intervals=[[0, 10], [10, 20]])
        result = retrieval_eval([q], lambda query, k: [{"start": 0, "end": 20}])
        self.assertEqual(result["summary"]["evidence_recall_at_5_iou_0_5"], 0.5)
        self.assertEqual(result["summary"]["multi_hop_success_at_5_iou_0_5"], 0.0)
        result = retrieval_eval([q], lambda query, k: [{"start": 0, "end": math.inf}])
        self.assertEqual(result["summary"]["retrieval_error_count"], 1)
        self.assertEqual(result["summary"]["hit_at_5_iou_0_5"], 0.0)


if __name__ == "__main__":
    unittest.main()
