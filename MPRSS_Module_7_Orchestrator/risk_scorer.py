"""
MPRSS Module 6 - Risk Scoring & Classification
Combines static permission score (Module 2), anomaly score (Module 3), and
behavioral score (Module 1 - not yet trained) into a single 0-100 risk
score, per the proposal's Section 3.1.5 spec:

    "The system shall generate a risk score between 0 and 100 for each
    analyzed APK. Risk scores shall combine static permission scores,
    behavioral suspicion scores, anomaly scores, and ML prediction
    confidence. The system shall categorize risk levels as Low (0-30),
    Medium (31-60), High (61-85), and Critical (86-100)."

HONEST DESIGN NOTE: Module 1 (behavioral RNN/LSTM) is not trained yet, so
there is no real behavioral SCORE - only raw event counts from Module 7's
Frida hooks. This combiner accepts behavioral_score=None and correctly
redistributes weight across whichever components ARE available, rather
than fabricating a 4-way blend that doesn't exist yet. Once Module 1
produces real scores, pass them in and the weighting adjusts automatically
with zero code changes needed elsewhere.

Default weights reflect each component's current reliability:
  - static (0.97 accuracy, fully trained Random Forest):        highest
  - anomaly (documented RBF-saturation limitation, see Module 3
    session notes - directional signal, not a precise probability): lower
  - behavioral (once Module 1 exists):                            moderate

These are a reasonable, documented starting point - NOT empirically
tuned against a labeled combined-score dataset (that would require
ground-truth labels across the full pipeline, which doesn't exist yet).
Worth flagging as a known limitation/future-work item, same as Module 3's
FPR/detection-rate honesty.
"""

DEFAULT_WEIGHTS = {
    'static': 0.5,
    'anomaly': 0.3,
    'behavioral': 0.2,
}


def get_risk_level(score):
    """Map a 0-100 score to the risk bands defined in the proposal."""
    if score <= 30:
        return 'Low'
    elif score <= 60:
        return 'Medium'
    elif score <= 85:
        return 'High'
    else:
        return 'Critical'


def compute_combined_risk(static_score, anomaly_score=None, behavioral_score=None,
                           static_confidence=None, weights=None):
    """
    Combine available risk signals into one 0-100 score.

    Parameters
    ----------
    static_score : float or None
        Module 2's Random Forest score (0-100). Required in practice -
        Module 2 always runs - but accepted as None for completeness/testing.
    anomaly_score : float or None
        Module 3's One-Class SVM score (0-100). None if no network flows
        were captured for this run.
    behavioral_score : float or None
        Module 1's trained score (0-100). Always None until Module 1 is
        built - this parameter exists so the function requires zero
        changes when that day comes.
    static_confidence : float or None
        Module 2's prediction confidence (0.0-1.0). Used as a modest
        adjustment: low confidence pulls the combined score slightly
        toward neutral (50), since an uncertain classification shouldn't
        be reported with the same weight as a confident one.
    weights : dict or None
        Override the default component weights. Must contain keys
        'static', 'anomaly', 'behavioral' (values don't need to sum to 1 -
        they're renormalized over whichever components are present).

    Returns
    -------
    dict with combined_score, risk_level, components_used, weights_used,
    and a human-readable explanation of how the score was built.
    """
    w = weights or DEFAULT_WEIGHTS

    components = {
        'static': static_score,
        'anomaly': anomaly_score,
        'behavioral': behavioral_score,
    }
    available = {k: v for k, v in components.items() if v is not None}

    if not available:
        return {
            'combined_score': None,
            'risk_level': None,
            'components_used': [],
            'weights_used': {},
            'explanation': 'No risk components available - cannot compute a score.',
        }

    # Renormalize weights over only the available components, so a missing
    # behavioral score (the normal case today) doesn't silently zero out
    # part of the score - static+anomaly's relative weighting is preserved.
    total_weight = sum(w[k] for k in available)
    normalized_weights = {k: w[k] / total_weight for k in available}

    raw_combined = sum(available[k] * normalized_weights[k] for k in available)

    # Confidence adjustment: pull toward neutral (50) when the static
    # model itself is uncertain. A confidence of 1.0 leaves the score
    # untouched; a confidence of 0.5 (coin-flip) pulls it 50% of the way
    # toward neutral. This only applies when static was actually used.
    if static_confidence is not None and 'static' in available:
        pull_factor = 1.0 - static_confidence
        combined = raw_combined + (50 - raw_combined) * pull_factor * 0.5
    else:
        combined = raw_combined

    combined = round(max(0, min(100, combined)), 2)

    weight_str = ', '.join(f"{k}={normalized_weights[k]:.0%}" for k in available)
    explanation = (
        f"Combined from {len(available)} of 3 possible components "
        f"({', '.join(available.keys())}), weighted {weight_str}."
    )
    if 'behavioral' not in available:
        explanation += (
            " Behavioral score not included - Module 1's RNN/LSTM is not "
            "trained yet; only raw event counts are available from Module 7."
        )

    return {
        'combined_score': combined,
        'risk_level': get_risk_level(combined),
        'components_used': list(available.keys()),
        'weights_used': normalized_weights,
        'explanation': explanation,
    }


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Quick manual sanity checks
    print("Static only (no anomaly, no behavioral):")
    print(compute_combined_risk(static_score=6, static_confidence=0.935))
    print()

    print("Static + anomaly (today's typical real case):")
    print(compute_combined_risk(static_score=6, anomaly_score=71.03, static_confidence=0.935))
    print()

    print("All three (future, once Module 1 exists):")
    print(compute_combined_risk(static_score=85, anomaly_score=90, behavioral_score=75, static_confidence=0.98))