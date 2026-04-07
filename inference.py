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


def rule_based_fallback(obs: Dict[str, Any], task_id: str = "medium") -> Dict[str, str]:
    """
    Smart rule-based fallback when LLM is unavailable.
    
    Meal costs/effects (approximate):
      burger : hunger -4, health -1, budget -30
      salad  : hunger -2, health +2, budget -10
      rice   : hunger -3, health  0, budget -20
    """
    hunger = obs.get("hunger", 10)
    health = obs.get("health", 5)
    budget = obs.get("budget", 50)

    if task_id == "easy":
        # Just kill hunger as fast as possible
        if hunger >= 4 and budget >= 30:
            return {"food": "burger"}
        elif hunger >= 3 and budget >= 20:
            return {"food": "rice"}
        else:
            return {"food": "salad"}

    elif task_id == "medium":
        # Balance hunger reduction and health
        if health <= 4:
            return {"food": "salad"}          # restore health
        if hunger >= 6 and budget >= 30:
            return {"food": "burger"}
        elif hunger >= 3:
            return {"food": "rice"}
        else:
            return {"food": "salad"}

    else:  # hard — budget matters most
        if budget < 20:
            return {"food": "salad"}          # cheapest option
        if health <= 3:
            return {"food": "salad"}
        if hunger >= 7 and budget >= 30 and health >= 5:
            return {"food": "burger"}
        elif hunger >= 3 and budget >= 20:
            return {"food": "rice"}
        else:
            return {"food": "salad"}


def choose_meal(obs: Dict[str, Any], task_id: str = "medium") -> Dict[str, str]:
    """Uses LLM proxy to decide on meal; falls back to rule-based on failure."""

    prompt = f"""You are an AI nutrition assistant helping optimize meals.

Current state:
- Hunger: {obs['hunger']}/10  (lower is better)
- Health: {obs['health']}/10  (higher is better)
- Budget: {obs['budget']}/100 (higher is better)
- Task: {task_id}

Meal options:
- burger: reduces hunger by ~4, costs ~30, slightly reduces health
- salad:  reduces hunger by ~2, costs ~10, improves health by ~2
- rice:   reduces hunger by ~3, costs ~20, neutral health

Choose the BEST single meal for this state. Reply with ONLY one word: burger, salad, or rice."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        choice = response.choices[0].message.content.strip().lower()

        if "burger" in choice:
            return {"food": "burger"}
        elif "salad" in choice:
            return {"food": "salad"}
        elif "rice" in choice:
            return {"food": "rice"}
        else:
            # LLM returned something unexpected — use fallback
            print(f"[WARN] Unexpected LLM response '{choice}', using rule-based fallback.")
            return rule_based_fallback(obs, task_id)

    except Exception as e:
        print(f"[WARN] LLM call failed: {e}. Using rule-based fallback.")
        return rule_based_fallback(obs, task_id)


def run_episode(task_id: str = "medium"):
    print(f"[START] Task: {task_id}")

    # Reset Environment
    try:
        res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
        res.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to reset environment: {e}")
        return 0.0

    data = res.json()
    obs = data["observation"]
    done = data.get("done", False)
    step_num = 1
    total_reward = 0.0

    while not done:
        # Agent decides (LLM with rule-based fallback)
        action_dict = choose_meal(obs, task_id)

        # Send action to environment
        try:
            res = requests.post(f"{ENV_URL}/step", json={"action": action_dict}, timeout=30)
            res.raise_for_status()
        except Exception as e:
            print(f"[ERROR] Step {step_num} failed: {e}")
            break

        data = res.json()
        obs = data["observation"]
        reward = data.get("reward", 0.0)
        done = data.get("done", False)

        total_reward += reward

        print(
            f"[STEP] {step_num}: {{"
            f"'action': {action_dict['food']}, "
            f"'hunger': {obs['hunger']}, "
            f"'budget': {obs['budget']}, "
            f"'health': {obs['health']}, "
            f"'reward': {round(reward, 2)}, "
            f"'done': {done}}}"
        )
        step_num += 1

        if step_num > 50:
            break

        time.sleep(0.01)

    print(f"[END] Total Reward: {round(total_reward, 2)}")
    return total_reward


if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    results = {}

    for task in tasks:
        print(f"\n{'='*40}")
        print(f"Evaluating Task: {task}")
        total_rew = run_episode(task)
        results[task] = round(total_rew, 2)
        print(f"Total Reward for '{task}': {results[task]}")

    print("\n======== FINAL RESULT ========")
    for task, rew in results.items():
        print(f"  {task}: {rew}")
    print("Inference Complete.")