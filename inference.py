import os
import json
import requests
import time
from typing import Dict, Any
from openai import OpenAI

# 🔑 Mandatory Hackathon Variables (injected by grader)
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "sk-noop")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# 🔌 Environment Server URL — MUST match Dockerfile port (7860)
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# 🧬 OpenAI Client (points at the Hackathon's LiteLLM proxy)
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
)

def extract_observation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize reset/step responses across OpenEnv wrapper versions."""
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

def fallback_policy(obs: Dict[str, Any]) -> Dict[str, str]:
    """Deterministic backup policy used when LLM call fails."""
    if obs["health"] <= 3:
        return {"food": "salad"}
    if obs["budget"] <= 20:
        return {"food": "rice"}
    if obs["hunger"] >= 8:
        return {"food": "rice"}
    return {"food": "salad"}


def choose_meal(obs: Dict[str, Any]) -> Dict[str, str]:
    """Uses LLM proxy to decide on meal based on current state."""
    prompt = f"""You are an AI nutrition assistant. Current state:
Hunger: {obs['hunger']}/10
Health: {obs['health']}/10
Budget: {obs['budget']}/100

Choose one action: burger, salad, or rice.
Return ONLY one word."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
    except Exception as e:
        print(f"[WARN] LLM request failed, using fallback policy: {str(e)}")
        return fallback_policy(obs)

    choice = response.choices[0].message.content.strip().lower()

    if "burger" in choice:
        return {"food": "burger"}
    elif "salad" in choice:
        return {"food": "salad"}
    else:
        return {"food": "rice"}

def run_episode(task_id: str = "medium"):
    print("[START]")

    # Reset Environment — NO try/except, let it crash loudly
    res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    res.raise_for_status()

    data = res.json()
    obs = extract_observation(data)
    done = data.get("done", False)
    step_num = 1
    total_reward = 0.0

    while not done:
        # Agent decides via LLM — NO try/except, let it crash loudly
        action_dict = choose_meal(obs)

        # Send action — NO try/except, let it crash loudly
        res = requests.post(f"{ENV_URL}/step", json={"action": action_dict})
        res.raise_for_status()

        data = res.json()
        obs = extract_observation(data)
        reward = data.get("reward", 0.0)
        done = data.get("done", False)

        total_reward += reward

        print(f"[STEP] {step_num}: {{'hunger': {obs['hunger']}, 'budget': {obs['budget']}, 'health': {obs['health']}, 'reward': {round(reward, 2)}, 'done': {done}}}")
        step_num += 1

        if step_num > 50:
            break

        time.sleep(0.01)

    print(f"[END] Total Reward: {round(total_reward, 2)}")
    return total_reward

if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        print(f"\nEvaluating Task: {task}")
        total_rew = run_episode(task)
        print(f"Total Reward: {round(total_rew, 2)}")

    print("\n======== FINAL RESULT ========")
    print("Baseline Inference Complete.")