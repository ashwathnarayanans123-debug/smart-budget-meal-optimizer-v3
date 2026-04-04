def grade_easy(final_state):
    # Scale: 0.0 – 1.0
    hunger = final_state.get("hunger", 10)
    return float(max(0, 10 - hunger)) / 10.0

def grade_medium(final_state):
    # Scale: 0.0 – 1.0
    hunger = final_state.get("hunger", 10)
    health = final_state.get("health", 0)
    hunger_score = max(0, 10 - hunger)
    return float((hunger_score + health) / 2) / 10.0

def grade_hard(final_state):
    # Scale: 0.0 – 1.0
    hunger = final_state.get("hunger", 10)
    health = final_state.get("health", 0)
    budget = final_state.get("budget", 0)
    
    hunger_score = max(0, 10 - hunger)
    budget_score = budget / 10 # 0–100 -> 0–10
    
    return float((hunger_score + health + budget_score) / 3) / 10.0