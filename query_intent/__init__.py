"""
query_intent — Ukrainian CS query intent classifier.

Classifies a user query as:
  0 = solution_oriented  (user wants code / a ready answer)
  1 = learning_oriented  (user wants to understand a concept)

Quick start
-----------
    from query_intent import QueryClassifier

    # load best model automatically
    clf = QueryClassifier.from_best(dataset="uk")

    # or load a specific checkpoint
    clf = QueryClassifier("results/uk__bert-base-multilingual-cased__ml256__lr2e-05__tsfull/best")

    clf("Що таке рекурсія?")
    # IntentResult(intent='learning_oriented', confidence=0.94) ████████████████████

    clf.predict_batch([
        "Напиши функцію сортування на Python",
        "Чим відрізняється list від tuple?",
    ])

Integration example (LMS / adaptive tutoring system)
-----------------------------------------------------
    clf = QueryClassifier.from_best(dataset="uk")

    def handle_student_query(query: str):
        result = clf(query)
        if result.intent == "learning_oriented":
            return give_hint()       # student wants to understand
        else:
            return give_solution()   # student wants a ready answer
"""

from .classifier import QueryClassifier, IntentResult, find_best_checkpoint

__all__ = ["QueryClassifier", "IntentResult", "find_best_checkpoint"]
__version__ = "0.1.0"
