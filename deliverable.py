"""
needle.vegr_weighting

Implements VEGR term weighting analysis and coverage weakness reporting.

This module provides:
- VEGR term weighting computation for a given document-term matrix.
- Coverage weakness analysis: identifies terms with low or zero coverage.
- Reporting of coverage weaknesses up front.

References:
- https://github.com/emrsyah/needle/issues/15

Author: emrsyah contributors
"""

from typing import List, Dict, Tuple, Any


def compute_vegr_weights(
    doc_term_matrix: List[List[int]],
    term_names: List[str]
) -> Dict[str, float]:
    """
    Compute VEGR (Variance-Entropy-Gain Ratio) weights for each term.

    Args:
        doc_term_matrix: List of documents, each as a list of term frequencies.
        term_names: List of term names corresponding to columns in doc_term_matrix.

    Returns:
        Dictionary mapping term name to its VEGR weight.
    """
    import math

    num_docs = len(doc_term_matrix)
    num_terms = len(term_names)
    weights: Dict[str, float] = {}

    # Transpose doc_term_matrix to term_doc_matrix for easier computation
    term_doc_matrix = [[doc[i] for doc in doc_term_matrix] for i in range(num_terms)]

    for idx, term in enumerate(term_names):
        freqs = term_doc_matrix[idx]
        mean = sum(freqs) / num_docs if num_docs > 0 else 0.0
        variance = sum((f - mean) ** 2 for f in freqs) / num_docs if num_docs > 0 else 0.0

        # Entropy calculation
        total = sum(freqs)
        if total == 0:
            entropy = 0.0
        else:
            probs = [f / total for f in freqs if f > 0]
            entropy = -sum(p * math.log2(p) for p in probs)

        # Gain Ratio (simplified for unsupervised context)
        # Here, we use variance / (1 + entropy) to avoid division by zero
        gain_ratio = variance / (1.0 + entropy)

        weights[term] = gain_ratio

    return weights


def analyse_coverage_weakness(
    doc_term_matrix: List[List[int]],
    term_names: List[str],
    coverage_threshold: float = 0.05
) -> List[Tuple[str, float]]:
    """
    Analyse coverage weakness for each term.

    Args:
        doc_term_matrix: List of documents, each as a list of term frequencies.
        term_names: List of term names corresponding to columns in doc_term_matrix.
        coverage_threshold: Minimum fraction of documents a term must appear in.

    Returns:
        List of tuples (term, coverage_fraction) for terms below threshold.
    """
    num_docs = len(doc_term_matrix)
    num_terms = len(term_names)
    weaknesses: List[Tuple[str, float]] = []

    for idx, term in enumerate(term_names):
        doc_count = sum(1 for doc in doc_term_matrix if doc[idx] > 0)
        coverage = doc_count / num_docs if num_docs > 0 else 0.0
        if coverage < coverage_threshold:
            weaknesses.append((term, coverage))

    return weaknesses


def report_coverage_weakness(
    weaknesses: List[Tuple[str, float]]
) -> str:
    """
    Generate a report string for coverage weaknesses.

    Args:
        weaknesses: List of tuples (term, coverage_fraction).

    Returns:
        String report listing weak terms and their coverage.
    """
    if not weaknesses:
        return "All terms meet the coverage threshold."

    lines = ["Coverage Weakness Report:"]
    for term, coverage in weaknesses:
        lines.append(f" - Term '{term}' has coverage: {coverage:.2%}")
    return "\n".join(lines)


# Example usage and test
def _test_vegr_weighting_and_coverage() -> None:
    """
    Test VEGR weighting and coverage weakness analysis.
    """
    # Example doc-term matrix (3 docs x 4 terms)
    doc_term_matrix = [
        [1, 0, 2, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 1]
    ]
    term_names = ["apple", "banana", "carrot", "date"]

    weights = compute_vegr_weights(doc_term_matrix, term_names)
    print("VEGR Weights:")
    for term, weight in weights.items():
        print(f"  {term}: {weight:.4f}")

    weaknesses = analyse_coverage_weakness(doc_term_matrix, term_names, coverage_threshold=0.5)
    report = report_coverage_weakness(weaknesses)
    print(report)


if __name__ == "__main__":
    _test_vegr_weighting_and_coverage()