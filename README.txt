# Deep Q-Learning Experiment

This project is a recreation of the seminal DeepMind experiment from their original Deep Q-Learning paper. The implementation trains a deep reinforcement learning agent to play Atari games using Q-learning with function approximation.

## Files Included
- **`main.py`**: Core implementation of the Deep Q-Learning algorithm, including training and evaluation of the agent.
- **`requirements.txt`**: List of Python dependencies needed to run the project.

## Getting Started
It is recommended to run the project in a virtual environment to avoid dependency conflicts.

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the experiment:
   ```bash
   python main.py
   ```

## Acknowledgments
This project is based on the work presented in the DeepMind paper: *"Playing Atari with Deep Reinforcement Learning."*
