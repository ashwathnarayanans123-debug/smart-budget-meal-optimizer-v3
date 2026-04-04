import os
import json
import requests
import time
from typing import Dict, Any
from openai import OpenAI

# 🔑 Mandatory Hackathon Variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo") # Or meta-llama/Llama-3.2-3B-Instruct
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

# 🧬 Initialization of the mandatory OpenAI Client
client = OpenAI(
    api_key=HF_TOKEN or "sk-noop",
    base_url=os.getenv("OPENAI_API_BASE", None) # Optional override
)

def choose_meal(obs: Dict[str, Any]) -> Dict[str, str]:
    """Uses LLM to decide on meal (burger, salad, rice) based on current state."""
    prompt = f"""
    You are an AI nutrition assistant. Current state:
    Hunger: {obs['hunger']}/10 (0 is best)
    Health: {obs['health']}/10 (10 is best)
    Budget: {obs['budget']}/100 (100 is best)

    Available actions: burger, salad, rice.
    Choose the best meal choice to optimize long-term health and budget while reducing hunger.
    Return ONLY JSON: {{"food": "burger" | "salad" | "rice"}}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        # Heuristic fallback for non-LLM testing
        return {"food": "rice"}


def run_episode(task_id: str = "medium"):
    print("[START]")
    
    # 🔌 Reset Environment
    res = requests.post(f"{API_BASE_URL}/reset", json={"task_id": task_id})
    if res.status_code != 200:
        return 0

    data = res.json()
    obs = data["observation"]
    done = data.get("done", False)
    step_num = 1
    total_reward = 0.0

    while not done:
        # 🧠 Agent decides
        action_dict = choose_meal(obs)
        
        # 🚀 Send Action to Environment
        payload = {"action": action_dict}
        res = requests.post(f"{API_BASE_URL}/step", json=payload)
        
        if res.status_code != 200:
            break
            
        data = res.json()
        obs = data["observation"]
        reward = data.get("reward", 0.0)
        done = data.get("done", False)
        
        total_reward += reward
        
        # 📊 Mandatory Structured Logs [STEP]
        print(f"[STEP] {step_num}: {{'hunger': {obs['hunger']}, 'budget': {obs['budget']}, 'health': {obs['health']}, 'reward': {round(reward, 2)}, 'done': {done}}}")
        step_num += 1
        
        if step_num > 50: break
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