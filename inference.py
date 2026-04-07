import os
import json
import requests
import time
from typing import Dict, Any
from openai import OpenAI

# 🔑 Mandatory Hackathon Variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "sk-noop")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

def find_env_url():
    candidates = [
        "http://127.0.0.1:8000",
        "http://127.0.0.1:7860", 
        "http://0.0.0.0:8000",
        "http://0.0.0.0:7860",
        "http://localhost:8000",
        "http://localhost:7860",
        "http://environment:8000",
        "http://server:8000"
    ]
    if os.getenv("OPENENV_BASE_URL"):
        candidates.insert(0, os.getenv("OPENENV_BASE_URL"))
        
    for url in candidates:
        try:
            # Send a fast HEAD request to check if port is open
            requests.head(url, timeout=1)
            print(f"✅ Found Environment server at: {url}")
            return url
        except requests.exceptions.ConnectionError:
            continue
    raise ConnectionError("Could not find Environment server on any candidate ports! Hackathon networking error.")

# 🔌 Dynamically locate the Environment Server port
ENV_URL = find_env_url()

# 🧬 Initialization of the mandatory OpenAI Client
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL
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
    Return only a single exact word from the actions: burger, salad, or rice. DO NOT return any other text.
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    choice = response.choices[0].message.content.strip().lower()
    
    # Simple keyword extraction in case it returns more than one word
    if "burger" in choice:
        return {"food": "burger"}
    elif "salad" in choice:
        return {"food": "salad"}
    else:
        return {"food": "rice"}


def run_episode(task_id: str = "medium"):
    print("[START]")
    
    # 🔌 Reset Environment
    res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    res.raise_for_status()

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
        res = requests.post(f"{ENV_URL}/step", json=payload)
        res.raise_for_status()
            
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