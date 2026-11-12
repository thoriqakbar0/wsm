"""Provider-free verification and deterministic summaries of retained campaigns."""

from collections import Counter, defaultdict
import math
import random
import statistics

from .benchmark import ARMS, STATUSES, campaign, digest, identity, read


def _evaluation(value):
    cases = value["cases"]
    if not cases:
        raise ValueError("Missing per-case grading evidence")
    for row in cases:
        if (type(row["target_nll"]) not in (int, float) or not math.isfinite(row["target_nll"])
                or row["target_nll"] < 0 or type(row["target_tokens"]) is not int or row["target_tokens"] < 1):
            raise ValueError("Invalid per-case grading evidence")
    mean = statistics.mean(row["target_nll"] for row in cases)
    if not math.isclose(mean, value["loss"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Grading aggregate disagrees with per-case evidence")
    tokens = sum(row["target_tokens"] for row in cases)
    weighted = sum(row["target_nll"] * row["target_tokens"] for row in cases) / tokens
    return {"nll": mean, "token_weighted_nll": weighted, "target_tokens": tokens, "cases": len(cases)}


def _quantile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize(pairs, *, seed, samples):
    """Resample tasks within each family, then paired repetitions within each task.

    Each observation is (improvement over unchanged, over script, regression).
    Families and tasks receive equal weight, regardless of repetition count.
    """
    if not pairs:
        return None
    families = sorted(pairs)

    def aggregate(groups):
        return [statistics.mean(statistics.mean(statistics.mean(r[i] for r in repetitions)
                    for repetitions in tasks) for tasks in groups) for i in range(3)]

    groups = [[pairs[family][task] for task in sorted(pairs[family])] for family in families]
    means = aggregate(groups)
    rng = random.Random(seed)
    draws = [[] for _ in range(3)]
    for _ in range(samples):
        sampled = []
        for tasks in groups:
            selected = rng.choices(tasks, k=len(tasks))
            sampled.append([rng.choices(repetitions, k=len(repetitions)) for repetitions in selected])
        for values, result in zip(draws, aggregate(sampled), strict=True):
            values.append(result)
    result = {}
    names = ("improvement_unchanged", "improvement_scripted", "regression")
    flat = [r for tasks in groups for repetitions in tasks for r in repetitions]
    for i, name in enumerate(names):
        values = [r[i] for r in flat]
        result[name] = {"mean": means[i], "median": statistics.median(values),
                        "min": min(values), "max": max(values),
                        "stddev": statistics.pstdev(values),
                        "ci95": [_quantile(draws[i], 0.025), _quantile(draws[i], 0.975)]}
    result["counts"] = {"families": len(groups), "tasks": sum(len(tasks) for tasks in groups),
                        "agent_repetitions": len(flat)}
    return result


def report(directory):
    root, manifest, config = campaign(directory)
    manifest_hash = identity(manifest)
    tasks = {t["id"]: t for t in manifest["tasks"]}
    rows, grades, outcomes = [], {}, {}
    reserved = config["budget"]["prior_reserved_usd"]
    for spec in manifest["schedule"]:
        folder = root / "trials" / spec["id"]
        outcome_path = folder / "outcome.json"
        if outcome_path.exists():
            outcome = read(outcome_path)
            if outcome["trial_id"] != spec["id"] or outcome["manifest_sha256"] != manifest_hash:
                raise ValueError("Outcome does not belong to this frozen schedule")
            if outcome["status"] not in STATUSES:
                raise ValueError("Invalid terminal outcome")
        else:
            outcome = {"status": "outcome_unknown" if (folder / "intent.json").exists() else "failed",
                       "reason": "missing_terminal_record"}
        row = {**spec, "family": tasks[spec["task"]]["family"], "status": outcome["status"],
               "reason": outcome.get("reason"), "task": spec["task"], "nll": None, "regression_nll": None,
               "seconds": outcome.get("seconds"), "reserved_usd": 0, "actual_usd": None,
               "usage_status": "unreconciled"}
        reservation = folder / "reservation.json"
        if reservation.exists():
            entry = read(reservation)
            amount = entry["usd"]
            if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
                raise ValueError("Invalid reservation")
            reserved += amount
            if not math.isclose(entry["cumulative_usd"], reserved, abs_tol=1e-10):
                raise ValueError("Budget reservation journal does not reconcile")
            row["reserved_usd"] = amount
        if row["status"] == "completed":
            receipt_path = root / "restricted" / spec["id"] / "grade.json"
            if digest(receipt_path) != outcome["grade_sha256"]:
                raise ValueError("Trusted grade receipt changed")
            receipt = read(receipt_path)
            frozen = read(folder / "frozen.json")
            if (receipt["trial_id"] != spec["id"] or receipt["manifest_sha256"] != manifest_hash
                    or receipt["checkpoint"] != outcome["checkpoint"]
                    or receipt["checkpoint"] != "sha256:" + frozen["sha256"]
                    or digest(root / frozen["weights"]) != frozen["sha256"] or not frozen["editor_closed"]):
                raise ValueError("Artifact or grade binding is invalid")
            for kind, grade in receipt["grades"].items():
                if grade["checkpoint"] != receipt["checkpoint"]:
                    raise ValueError("Grader checkpoint differs from frozen artifact")
                _evaluation(grade["baseline"])
                _evaluation(grade["final"])
                expected = tasks[spec["task"]]["private"]["count"] if kind == "task" else manifest["regression"]["count"]
                if len(grade["final"]["cases"]) != expected:
                    raise ValueError("Grading case count differs from frozen partition")
            task_grade = _evaluation(receipt["grades"]["task"]["final"])
            regression_grade = _evaluation(receipt["grades"]["regression"]["final"])
            row.update(nll=task_grade["nll"], token_weighted_nll=task_grade["token_weighted_nll"],
                       target_tokens=task_grade["target_tokens"], case_count=task_grade["cases"],
                       regression_nll=regression_grade["nll"], checkpoint=receipt["checkpoint"])
            grades[spec["id"]] = receipt["grades"]
        rows.append(row)
        outcomes[spec["id"]] = outcome
    if reserved > config["budget"]["max_usd"] + 1e-10:
        raise ValueError("Campaign reservations exceed allocation")
    baselines = {(row["task"], row["arm"], row["repetition"]): row for row in rows
                 if row["arm"] != "agent" and row["status"] == "completed"}
    pairs = defaultdict(dict)
    thresholds = manifest["thresholds"]
    tolerance = (thresholds or {}).get("numeric_tolerance", 0.0)
    successes = 0
    for row in rows:
        row["successful_edit"] = None
        if row["status"] != "completed":
            continue
        unchanged = baselines.get((row["task"], "unchanged", 0))
        if unchanged:
            for kind in ("task", "regression"):
                expected = grades[unchanged["id"]][kind]["baseline"]["cases"]
                actual = grades[row["id"]][kind]["baseline"]["cases"]
                if any(abs(a["target_nll"] - b["target_nll"]) > tolerance
                       or a["target_tokens"] != b["target_tokens"]
                       for a, b in zip(actual, expected, strict=True)):
                    raise ValueError("Trials did not share the same pristine evaluation baseline")
        if row["arm"] != "agent":
            continue
        script_repetition = row["repetition"] if tasks[row["task"]].get("scripted_stochastic", False) else 0
        scripted = baselines.get((row["task"], "scripted", script_repetition))
        if unchanged:
            row.update(improvement_unchanged=unchanged["nll"] - row["nll"],
                       regression=row["regression_nll"] - unchanged["regression_nll"])
            if thresholds is not None:
                row["successful_edit"] = (row["improvement_unchanged"] >= thresholds["improvement"]
                                          and row["regression"] <= thresholds["regression"])
                successes += row["successful_edit"]
        if unchanged and scripted:
            row["improvement_scripted"] = scripted["nll"] - row["nll"]
            pairs[row["family"]].setdefault(row["task"], []).append(
                (row["improvement_unchanged"], row["improvement_scripted"], row["regression"]))
    stats = manifest["statistics"]
    summary = summarize(pairs, seed=stats["seed"], samples=stats["bootstrap_samples"])
    complete = all(row["status"] == "completed" for row in rows)
    decision = "inconclusive"
    if not manifest["confirmatory"]:
        decision = "exploratory"
    elif complete and summary:
        lower = [summary[name]["ci95"][0] for name in ("improvement_unchanged", "improvement_scripted")]
        upper = [summary[name]["ci95"][1] for name in ("improvement_unchanged", "improvement_scripted")]
        regression = summary["regression"]["ci95"]
        if min(lower) > thresholds["improvement"] and regression[1] <= thresholds["regression"]:
            decision = "positive"
        elif min(upper) < thresholds["improvement"] or regression[0] > thresholds["regression"]:
            decision = "negative"
    scheduled_agents = sum(row["arm"] == "agent" for row in rows)
    counts = Counter(row["status"] for row in rows)
    return {"format": "wsm.benchmark-report.v1", "name": manifest["name"], "manifest_sha256": manifest_hash,
            "synthetic": manifest["synthetic"], "decision": decision, "thresholds": thresholds,
            "complete": complete, "counts": {key: counts[key] for key in STATUSES},
            "rows": rows, "summary": summary,
            "families": {family: summarize({family: tasks}, seed=stats["seed"], samples=stats["bootstrap_samples"])
                         for family, tasks in sorted(pairs.items())},
            "successes": successes if thresholds is not None else None,
            "scheduled_agent_trials": scheduled_agents,
            "success_rate": successes / scheduled_agents if thresholds is not None else None,
            "cost": {"reserved_usd": reserved, "actual_usd": None, "billing": "unreconciled",
                     "per_success_reserved_usd": reserved / successes if successes else None,
                     "note": "thresholds missing" if thresholds is None else "no successes" if not successes else "reservation, not invoice"},
            "statistics": {**stats, "method": "hierarchical paired bootstrap; equal families and tasks",
                           "summary_scope": "completed paired trials only", "median_spread_scope": "completed paired repetitions",
                           "warning": "Small task samples yield exploratory intervals; repeated cases are not independent tasks"}}


def markdown(result):
    lines = ["# WSM benchmark report", "", f"Decision: **{result['decision']}**.", "",
             f"Synthetic fixture: {str(result['synthetic']).lower()}. Complete: {str(result['complete']).lower()}.", "",
             f"Manifest: `{result['manifest_sha256']}`.", "",
             f"Thresholds: `{result['thresholds']}`.", "",
             f"Terminal counts: `{result['counts']}`.", "",
             f"Successful edits: {result['successes']} of {result['scheduled_agent_trials']} scheduled agent trials.", "",
             f"Reserved USD: {result['cost']['reserved_usd']:.8g}. Actual billing: unreconciled. {result['cost']['note']}.", ""]
    if result["summary"]:
        lines.extend(["| Comparison | Mean | 95% interval | Median | Range |", "| --- | --- | --- | --- | --- |"])
        for name in ("improvement_unchanged", "improvement_scripted", "regression"):
            values = result["summary"][name]
            lines.append(f"| {name} | {values['mean']:.8g} | {values['ci95']} | {values['median']:.8g} | {values['min']:.8g} to {values['max']:.8g} |")
        lines.extend(["", f"Sample counts: `{result['summary']['counts']}`.", ""])
    lines.extend([
             "| Trial | Status | Task NLL | Regression NLL | Reserved USD |", "| --- | --- | --- | --- | --- |"]
    )
    for row in result["rows"]:
        def number(value):
            return "missing" if value is None else f"{value:.8g}"
        lines.append(f"| {row['id']} | {row['status']} | {number(row['nll'])} | {number(row['regression_nll'])} | {number(row['reserved_usd'])} |")
    lines.extend(["", "Private case text and per-case grades are omitted. Costs are reservations, not invoices.", "",
                  "Summary statistics use completed paired trials. Failures remain in the scheduled success denominator.", ""])
    return "\n".join(lines)
