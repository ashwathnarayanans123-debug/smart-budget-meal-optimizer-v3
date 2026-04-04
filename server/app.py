import os
import uvicorn
from fastapi.responses import FileResponse, JSONResponse
from openenv.core.env_server.http_server import create_app
from fastapi.middleware.cors import CORSMiddleware
from server.environment import MealEnv
from models import TicketAction, TicketObservation, TicketReward
from dotenv import load_dotenv

# Load local .env
load_dotenv()

def get_env():
    return MealEnv()

# Create the standard OpenEnv app
app = create_app(
    get_env,
    action_cls=TicketAction,
    observation_cls=TicketObservation,
)

# 🌍 Enable CORS for Hugging Face support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🌐 UI Support
static_dir = os.path.join(os.path.dirname(__file__), "static")
@app.get("/", include_in_schema=False)
async def read_index():
    path = os.path.join(static_dir, "index.html")
    if os.path.exists(path): return FileResponse(path)
    return JSONResponse({"error": "UI index.html missing!"})

@app.get("/state", include_in_schema=False)
async def get_state():
    # Retrieve current state from the environment
    state = app.state.env.state
    return JSONResponse({"observation": state.dict()})

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()