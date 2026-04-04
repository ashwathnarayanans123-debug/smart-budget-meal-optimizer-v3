from typing import Optional, Literal
from openenv.core.env_server.types import Action, Observation, BaseModel
from pydantic import Field

class TicketAction(Action): 
    food: Literal["burger", "salad", "rice"]


class TicketObservation(Observation):
    hunger: int = Field(ge=0, le=10)
    budget: int = Field(ge=0, le=100)
    health: int = Field(ge=0, le=10)


class TicketReward(BaseModel):
    reward: float
    done: bool
    observation: TicketObservation
    info: Optional[dict] = Field(default_factory=dict)