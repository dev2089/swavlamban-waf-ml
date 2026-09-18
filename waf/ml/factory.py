from waf.ml.ensemble import Phase4MLEnsemble


def trained_models() -> Phase4MLEnsemble:
    """Compatibility factory for callers that need a trained Phase 4 ensemble."""
    return Phase4MLEnsemble.train_default()
