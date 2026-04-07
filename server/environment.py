from typing import Optional
from openenv.core.env_server.interfaces import Environment
from models import TicketAction, TicketObservation, TicketReward

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

    def __init__(self):
        super().__init__()
        # Instance-level state avoids cross-request/session interference.
        self._hunger = 10
        self._budget = 80
        self._health = 8
        self._task_id = "medium"
        self._done = False
        self._step_count = 0
        self._max_steps = 12

    def _get_observation(self) -> TicketObservation:
        return TicketObservation(
            hunger=self._hunger,
            budget=self._budget,
            health=self._health,
        )

    def reset(self, task_id: Optional[str] = None):
        """Initializes a new scenario based on task constraint."""
        self._task_id = task_id or "medium"
        self._step_count = 0

        # 🎮 Scenario Initialization (Task-specific logic)
        if self._task_id == "easy":
            self._hunger = 10
            self._budget = 100
            self._health = 10
        elif self._task_id == "hard":
            self._hunger = 10
            self._budget = 50
            self._health = 5
        else:
            self._hunger = 10
            self._budget = 80
            self._health = 8

        self._done = False
        return self._get_observation()
    
    def step(self, action: TicketAction) -> TicketReward:
        food = action.food.lower()
        if self._done:
            return TicketReward(
                reward=0.0,
                done=True,
                observation=self._get_observation(),
                info={"task": self._task_id, "reason": "episode_already_done"},
            )

        self._step_count += 1

        # 🧪 Food Effects (Real-World Logic as per spec)
        if food == "burger":
            self._hunger -= 3
            self._budget -= 30
            self._health -= 1
        elif food == "salad":
            self._hunger -= 2
            self._budget -= 20
            self._health += 2
        elif food == "rice":
            self._hunger -= 4
            self._budget -= 15
            self._health += 1
        else:  # ❌ Invalid action penalty
            self._health -= 2

        # Living pressure: each decision step has slight metabolic drag.
        self._hunger += 1
        self._health -= 1 if food == "burger" else 0

        # 📉 Clamp Values
        self._hunger = max(0, min(10, self._hunger))
        self._budget = max(0, min(100, self._budget))
        self._health = max(0, min(10, self._health))

        # 🛑 Termination Logic
        self._done = (
            self._hunger == 0
            or self._budget <= 0
            or self._health <= 0
            or self._step_count >= self._max_steps
        )

        # 🎯 Reward Function (Meaningful Signal Trajectory)
        reward = (
            (10 - self._hunger) * 0.5
            + self._health * 0.3
            + self._budget * 0.1
        )

        if self._budget == 0 or self._health == 0:
            reward -= 8.0
        if self._step_count >= self._max_steps and self._hunger > 0:
            reward -= 4.0

        return TicketReward(
            reward=reward,
            done=self._done,
            observation=self._get_observation(),
            info={"task": self._task_id, "step_count": self._step_count},
        )
    
    @property
    def state(self):
        return self._get_observation()
