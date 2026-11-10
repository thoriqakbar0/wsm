"""A deterministic tensor edit for checking the SDK, without an efficacy claim."""


def edit(model):
    weights = model.weights["model.norm.weight"]
    weights[:16].mul_(0.95)
