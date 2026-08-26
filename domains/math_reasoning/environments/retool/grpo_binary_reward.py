from verl.utils.reward_score import math_dapo


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    """Strict boxed-answer reward with binary 0/1 scoring.

    This intentionally does not include the ReTool recipe's turn/tool-use
    shaping. Tool use should only help through final answer correctness.
    """
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
    result["score"] = 1.0 if float(result.get("score", 0.0)) > 0 else 0.0
    result["acc"] = bool(result["score"])
    if result.get("pred") is None:
        result["pred"] = ""
    return result
