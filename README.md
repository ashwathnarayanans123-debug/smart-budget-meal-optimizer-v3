---
title: Smart Budget Meal Opt V2
emoji: 🥗
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# 🥗 Smart Budget Meal Optimizer

**Smart Budget Meal Optimizer** is a real-world AI environment that simulates food decision-making under constraints of hunger, budget, and health. 

Users interact with the system by selecting food options, and the environment responds with updated states and rewards based on the quality of decisions. The system models real-life challenges such as maintaining a healthy diet while managing limited financial resources.

---

## 🌎 1. Real-World Use Case
### 🍽️ What this represents
This is NOT just a game. It simulates a **real-life decision problem**: *“What should I eat with limited money while staying healthy?”*

### 💡 Potential Applications:
1.  **🥗 Personal Diet Planning**: Suggest meals based on budget + health constraints.
2.  **💰 Budget Optimization**: Helps users avoid overspending on food while maintaining nutrition.
3.  **🏥 Health-aware Decisions**: Encourages better food choices through quantified feedback.
4.  **🤖 AI Training Environment**: A robust platform for training agents in multi-objective optimization.

---

## 🧱 2. Environment Specification

### 📊 Observation Space
-   **hunger**: `0–10` (Target: 0)
-   **budget**: `0–100` (₹ value)
-   **health**: `0–10` (Target: 10)

### 🎮 Action Space
Choose a meal each step:
-   **🥗 Salad**: Healthy choice, low cost.
-   **🍔 Burger**: High hunger reduction, expensive, less healthy.
-   **🍚 Rice**: Balanced option for cost and health.

---

## 🧪 3. Tasks & Graders
-   **Task 1: Easy (Hunger Reduction)**: Focus on reducing hunger efficiently.
-   **Task 2: Medium (Balance Hunger + Health)**: Minimizing hunger without sacrificing health.
-   **Task 3: Hard (Full Constraint Optimization)**: Optimal long-term health, hunger, and budget balance.

---

## 🎮 4. How to Use the App

### ⚙ Installation
```bash
pip install -r requirements.txt
```

### 🚀 Running the App
1.  **Start the Server**:
    ```bash
    python server/app.py
    ```
2.  **Open the UI**:
    Open `http://127.0.0.1:8000` in your browser.

### 👤 User Instructions
1.  **Initialize**: The game starts with a reset state.
2.  **Choose Food**: Select **Salad**, **Burger**, or **Rice** based on your current stats.
3.  **See Results**: UI updates instantly with new stats and a reward signal.
4.  **Goal**: Maximize health and minimize hunger while staying within budget.
5.  **Game End**: The simulation ends if budget or health reaching 0.

---

## 📉 5. AI Extension & Requirements
This environment follows the **OpenEnv** specification, making it suitable for evaluating AI agents.

### 🔑 Mandatory Environment Variables
| Variable | Description |
| --- | --- |
| `API_BASE_URL` | The API endpoint for the LLM (Default: http://127.0.0.1:8000) |
| `MODEL_NAME` | The model identifier (e.g. `meta-llama/Llama-3.2-3B-Instruct`) |
| `HF_TOKEN` | Your Hugging Face / API key |

### 🤖 Running the Baseline Agent
Ensure you have the environment variables set, then run:
```bash
python inference.py
```

### 📊 Expected Baseline Results (0.0 - 1.0)
- **Easy Task**: 0.95 - 1.0
- **Medium Task**: 0.85 - 0.95
- **Hard Task**: 0.70 - 0.85

*"This environment can be extended to real-world applications such as meal planning apps, health recommendation systems, and financial budgeting assistants."*
