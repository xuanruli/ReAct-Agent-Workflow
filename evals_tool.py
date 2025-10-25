import functools
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from prompt import EVAL_LABEL_PROMPT, EVAL_SCORE_PROMPT
from pydantic import BaseModel, Field


def retry(*, retries: int = 3, backoff_sec: float = 3, tag: Optional[str] = None):
    """Decorator factory to retry a function with backoff and logging."""

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            last_err: Optional[BaseException] = None
            for attempt in range(1, retries + 1):
                try:
                    if attempt > 1:
                        print(f"[Retry] Attempt {attempt}/{retries} for {tag or func.__name__}...")
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        print(f"[Retry] Success on attempt {attempt} for {tag or func.__name__}")
                    return result
                except Exception as e:
                    last_err = e
                    if attempt >= retries:
                        print(f"[Retry] Failed after {retries} attempts for {tag or func.__name__}: {e}")
                        assert last_err is not None
                        raise RuntimeError("retry used all attempts") from last_err
                    delay = backoff_sec * attempt if attempt < retries else 60
                    print(
                        f"[Retry] Error on attempt {attempt}/{retries} "
                        f"for {tag or func.__name__}: {e}. Sleeping {delay:.1f}s"
                    )
                    time.sleep(delay)
            return None

        return _wrapper

    return _decorator


def format_strategy_block(strategies: List[Dict[str, Any]]) -> str:
    """Render a list of Strategy dicts in a consistent schema."""
    lines: List[str] = []
    for s in strategies:
        name = s.get("strategy_name", "unknown")
        desc = s.get("strategy_description", "")
        # formula = s.get("hook_selection_formula", "")
        # emotion = s.get("audience_emotion", "")
        # pacing = s.get("pacing_structure", "")
        lines.append(
            "- strategy_name: "
            + str(name)
            + "\n  strategy_description: "
            + str(desc)
            # + "\n  hook_selection_formula: "
            # + str(formula)
            # + "\n  audience_emotion: "
            # + str(emotion)
            # + "\n  pacing_structure: "
            # + str(pacing)
        )
    return "\n".join(lines)


class StrategyLabel(BaseModel):
    strategy_name: str = Field(..., description="exact strategy name in strategy set")
    rationale: str = Field(..., description="labelling rationale")


class StrategyScore(BaseModel):
    score: float
    rationale: str


EvalResult = StrategyLabel | StrategyScore


def llm_json_call(
    client: OpenAI, model: str, prompt: str, temperature: float, structure_output: EvalResult
) -> Dict[str, Any]:
    @retry(tag=f"chat:{model}")
    def _call() -> Dict[str, Any]:
        resp = client.responses.parse(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            text_format=structure_output,
            temperature=temperature,
        )
        structure = resp.output_parsed
        return structure.model_dump()

    return _call()


def eval_labeling(
    strategies: List[Dict[str, Any]],
    df: pd.DataFrame,
    transcript_col: str,
    client: OpenAI,
    model_name: str,
    temperature: float,
) -> List[Dict[str, Any]]:
    strategies_block = format_strategy_block(strategies)
    results: List[Dict[str, Any]] = []
    total = len(df)
    print(f"[Labeling] Start - rows={total}, model={model_name}, temperature={temperature}")

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        clip_id = row["clip_id"]
        transcript = str(row[transcript_col])
        hook = str(row["hook"])
        prompt = EVAL_LABEL_PROMPT.format(strategies_block=strategies_block, hook=hook, transcript=transcript)
        out = llm_json_call(client, model_name, prompt, temperature, StrategyLabel)
        label = out.get("strategy_name", "")
        rationale = out.get("rationale", "")
        results.append({"clip_id": clip_id, "hook": hook, "label": label, "rationale_label": rationale})
        print(f"[Labeling] {idx}/{total} clip_id={clip_id} label={label}")
    print("[Labeling] Done")
    pd.DataFrame(results).to_csv("eval_labeling_results.csv", index=False)
    return results


def eval_scoring(
    strategies: List[Dict[str, Any]],
    df_with_labels: pd.DataFrame,
    transcript_col: str,
    label_col: str,
    client: OpenAI,
    model_name: str,
    temperature: float,
) -> List[Dict[str, Any]]:
    strategies_block = format_strategy_block(strategies)
    results: List[Dict[str, Any]] = []
    total = len(df_with_labels)
    print(f"[Scoring] Start - rows={total}, model={model_name}, temperature={temperature}")

    for idx, (_, row) in enumerate(df_with_labels.iterrows(), start=1):
        project_id = str(row["project_id_long"])
        clip_id = str(row["clip_id"])
        short_video_link = str(row["short_video_link"])
        long_video_link = str(row["long_video_link"])
        transcript = str(row[transcript_col])
        hook = str(row["hook"])
        label = str(row[label_col])
        prompt = EVAL_SCORE_PROMPT.format(strategies_block=strategies_block, hook=hook, transcript=transcript, label=label)
        out = llm_json_call(client, model_name, prompt, temperature, StrategyScore)
        score = out.get("score", 0)
        rationale = out.get("rationale", "")
        results.append(
            {"clip_id": clip_id, "project_id": project_id, "short_video_link": short_video_link, "long_video_link": long_video_link, "hook": hook, "label": label, "score": score, "rationale_score": rationale}
        )
        print(f"[Score] {idx}/{total} clip_id={clip_id} label={label}")
    print("[Scoring] Done")
    return results


def evaluate(
    strategies: List[Dict[str, Any]],
    csv_path: str,
    transcript_col: str = "transcripts",
    model: str = "gpt-4.1",
    temperature: float = 0.1,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run labeling + scoring and return aggregate metrics"""
    df = pd.read_csv(csv_path)
    client = OpenAI()

    print("[API] Evaluate - labeling...")
    labeled = eval_labeling(strategies, df, transcript_col, client, model, temperature)
    result = df.copy()
    result["pred_label"] = [x["label"] for x in labeled]

    print("[API] Evaluate - scoring...")
    scored = eval_scoring(strategies, result, transcript_col, "pred_label", client, model, temperature)
    scores = [float(x.get("score", 0)) for x in scored]
    result["score"] = scores

    # distribution
    buckets = {1.0: 0, 0.5: 0, 0.0: 0}
    c = Counter(scores)
    for k in buckets:
        buckets[k] = int(c.get(k, 0))

    n = len(scores)
    total = float(sum(scores))
    avg = (total / n) if n else 0.0

    pred_dist = Counter(result["pred_label"]) if len(result) else {}

    metrics = {
        "num_samples": n,
        "score_counts": {"1": buckets[1.0], "0.5": buckets[0.5], "0": buckets[0.0]},
        "score_pct": {
            "1": (buckets[1.0] / n) if n else 0.0,
            "0.5": (buckets[0.5] / n) if n else 0.0,
            "0": (buckets[0.0] / n) if n else 0.0,
        },
        "average_score": avg,
        "weighted_total": total,
        "pred_label_distribution": dict(pred_dist),
    }
    print("[API] Evaluate - done:", metrics)
    return metrics, result
