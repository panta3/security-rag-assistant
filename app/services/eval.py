"""Eval harness — this is what makes the project defensible, don't skip it.

Plan: hand-build ~20-30 (question, expected_source_chunk) pairs from the
ingested docs. For each:
  - retrieval precision: did the true chunk show up in top-k?
  - citation accuracy: does the generated answer cite a real chunk that
    actually supports the claim, or a fabricated/irrelevant one?
  - hallucination rate: fraction of answers making claims not
    supported by any retrieved chunk
"""


def evaluate(eval_set_path: str) -> dict:
    # TODO: load eval_set (json), run each question through answer_query(),
    # score against expected sources, return aggregate metrics.
    raise NotImplementedError
