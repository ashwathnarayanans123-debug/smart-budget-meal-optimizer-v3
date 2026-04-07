import os
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# Mandatory variables from the hackathon checker.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct").strip()
API_KEY = (os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "").strip()

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860").strip()
BENCHMARK = os.getenv("BENCHMARK", "meal-optimization").strip()
MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


def validate_runtime_env() -> None:
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not API_KEY:
        missing.append("HF_TOKEN (or API_KEY)")
    if missing:
        raise RuntimeError("Missing required environment variable(s): " + ", ".join(missing))


def extract_observation(payload: Dict[str, Any]) -> Dict[str, int]:
    obs = payload.get("observation", payload)
    if isinstance(obs, dict) and "observation" in obs and "hunger" not in obs:
        inner = obs.get("observation")
        if isinstance(inner, dict):
            obs = inner
    return {
        "hunger": int(obs.get("hunger", 10)),
        "budget": int(obs.get("budget", 100)),
        "health": int(obs.get("health", 10)),
    }


def normalize_choice(text: str) -> str:
    value = (text or "").strip().lower()
    if "burger" in value:
        return "burger"
    if "salad" in value:
        return "salad"
    return "rice"


def fallback_policy(obs: Dict[str, int]) -> str:
    if obs["health"] <= 3:
        return "salad"
    if obs["budget"] <= 20 or obs["hunger"] >= 8:
        return "rice"
    return "salad"


def choose_action(obs: Dict[str, int]) -> Tuple[str, Optional[str]]:
    prompt = (
        "You are optimizing hunger, health, and budget.\n"
        f"Hunger: {obs['hunger']}/10\n"
        f"Health: {obs['health']}/10\n"
        f"Budget: {obs['budget']}/100\n"
        "Choose one action exactly: burger, salad, or rice.\n"
        "Return only the action word."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8,
            stream=False,
        )
        content = completion.choices[0].message.content or ""
        return normalize_choice(content), None
    except Exception as exc:
        return fallback_policy(obs), str(exc)


def format_error(value: Optional[str]) -> str:
    if not value:
        return "null"
    return value.replace("\n", " ").replace("\r", " ")


def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={format_error(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def run_episode(task_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task_id)

    try:
        res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        res.raise_for_status()
        payload = res.json()
        obs = extract_observation(payload)
        done = bool(payload.get("done", False))

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action, llm_error = choose_action(obs)
            res = requests.post(f"{ENV_URL}/step", json={"action": {"food": action}}, timeout=30)
            res.raise_for_status()
            payload = res.json()
            obs = extract_observation(payload)
            reward = float(payload.get("reward", 0.0))
            done = bool(payload.get("done", False))
            info = payload.get("observation", {}).get("info", {})
            env_error = None
            if isinstance(info, dict):
                env_error = info.get("last_action_error")
            error = env_error or llm_error

            rewards.append(reward)
            steps_taken = step
            log_step(step, action, reward, done, error)

        if rewards:
            # Normalize per-task score into [0,1] using observed reward range.
            min_r = min(rewards)
            max_r = max(rewards)
            total = sum(rewards)
            min_total = min_r * len(rewards)
            max_total = max_r * len(rewards)
            if max_total > min_total:
                score = (total - min_total) / (max_total - min_total)
            else:
                score = 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD
        return score
    finally:
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    validate_runtime_env()
    for task_name in ("easy", "medium", "hard"):
        run_episode(task_name)