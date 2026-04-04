import random
from typing import Optional, List, Dict
from openenv.core.env_server.interfaces import Environment
from models import TicketAction, TicketObservation, TicketReward
from tasks import TASKS

class MealEnv(Environment):
    """
    🧬 Smart Budget Meal Optimization Environment
    An environment where an agent balances Hunger, Health, and Budget.
    """
    # 🧪 metadata for OpenEnv framework
    metadata = {
        "name": "Smart Budget Meal Optimization Environment",
        "description": "Balancing health, budget, and hunger in a real-world nutrition simulation.",
        "version": "1.0.0",
        "author": "Ashwath1107"
    }

    # 🧬 Class-level state for stateful HTTP simulation
    _hunger = 10
    _budget = 100
    _health = 10
    _task_id = "medium"
    _done = False
    _rng = random.Random(42)

    def __init__(self):
        super().__init__()

    def _get_observation(self) -> TicketObservation:
        return TicketObservation(
            hunger=MealEnv._hunger,
            budget=MealEnv._budget,
            health=MealEnv._health,
        )

    def reset(self, task_id: Optional[str] = None):
        """Initializes a new scenario based on task constraint."""
        MealEnv._task_id = task_id or "medium"
        
        # 🎮 Scenario Initialization (Task-specific logic)
        if task_id == "easy":
            MealEnv._hunger = 10
            MealEnv._budget = 100
            MealEnv._health = 10
        elif task_id == "hard":
            MealEnv._hunger = 10
            MealEnv._budget = 50 
            MealEnv._health = 5 
        else:
            MealEnv._hunger = 10
            MealEnv._budget = 80
            MealEnv._health = 8
            
        MealEnv._done = False
        return self._get_observation()
    
    def step(self, action: TicketAction) -> TicketReward:
        food = action.food.lower()
        
        # 🧪 Food Effects (Real-World Logic as per spec)
        if food == "burger":
            MealEnv._hunger -= 3
            MealEnv._budget -= 30
            MealEnv._health -= 1
        elif food == "salad":
            MealEnv._hunger -= 2
            MealEnv._budget -= 20
            MealEnv._health += 2
        elif food == "rice":
            MealEnv._hunger -= 4
            MealEnv._budget -= 15
            MealEnv._health += 1
        else: # ❌ Invalid action penalty
            MealEnv._health -= 2
        
        # 📉 Clamp Values
        MealEnv._hunger = max(0, min(10, MealEnv._hunger))
        MealEnv._budget = max(0, min(100, MealEnv._budget))
        MealEnv._health = max(0, min(10, MealEnv._health))
        
        # 🛑 Termination Logic
        MealEnv._done = (
            MealEnv._hunger == 0 or
            MealEnv._budget <= 0 or
            MealEnv._health <= 0
        )
        
        # 🎯 Reward Function (Meaningful Signal Trajectory)
        reward = (
            (10 - MealEnv._hunger) * 0.5 +   # Hungrier = Bad
            MealEnv._health * 0.3 +          # Healthier = Good
            MealEnv._budget * 0.1            # Richer = Good
        )
        
        if MealEnv._done:
            # Penalty for failure states, or lower reward for premature finish
            if MealEnv._hunger > 0: reward -= 5
        
        return TicketReward(
            reward=reward,
            done=MealEnv._done,
            observation=self._get_observation(),
            info={"task": MealEnv._task_id}
        )
    
    @property
    def state(self):
        return self._get_observation()
