import numpy as np
import xgboost as xgb
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import os
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
import joblib

class ProjectAgent:
    def __init__(self):
        self.n_actions = 4
        self.models = []
        self.scalers = [StandardScaler() for _ in range(self.n_actions)]
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'max_depth': 6,
            'eta': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 3,
            'gamma': 0.1,
            'lambda': 1.5,
            'alpha': 0.5,
            'tree_method': 'hist',
            'max_leaves': 64,
            'seed': 42
        }

        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.9

        # Training parameters
        self.gamma = 0.99
        self.reward_scale = 1e-5
        
    def collect_episodes(self, env, n_steps):
        transitions = []
        obs, _ = env.reset()
        
        for _ in range(n_steps):
            action = self.act(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            transitions.append((obs, action, reward * self.reward_scale, next_obs, done or truncated))
            
            if done or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs
                
        return transitions

    def train_fqi(self, n_epochs: int = 100, steps_per_epoch = 100):
        env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)
        best_eval_reward = float('-inf')
        
        # Initialize models
        self.models = [
            xgb.XGBRegressor(**self.xgb_params) for _ in range(self.n_actions)
        ]
        
        print("Starting training...")
        transitions = []
        
        # Training loop
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            
            # Collect new transitions
            new_transitions = self.collect_episodes(env, steps_per_epoch)
            transitions.extend(new_transitions)
            
            # Prepare training data
            states = np.array([t[0] for t in transitions])
            actions = np.array([t[1] for t in transitions])
            rewards = np.array([t[2] for t in transitions])
            next_states = np.array([t[3] for t in transitions])
            dones = np.array([t[4] for t in transitions])
            
            next_q_values = []
            for action in range(self.n_actions):
                scaled_next_states = self.scalers[action].fit_transform(next_states)
                if epoch == 0:
                    next_q_values.append(np.zeros(len(transitions)))
                else:
                    next_q_values.append(self.models[action].predict(scaled_next_states))
            next_q_values = np.array(next_q_values).T
            
            max_next_q = np.max(next_q_values, axis=1)
            targets = rewards + self.gamma * (1 - dones) * max_next_q

            for action in range(self.n_actions):
                action_mask = actions == action
                if np.any(action_mask):
                    states_action = states[action_mask]
                    scaled_states = self.scalers[action].fit_transform(states_action)
                    self.models[action].fit(
                        scaled_states,
                        targets[action_mask],
                        verbose=False
                    )
            
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            eval_reward = self.evaluate(env, 10)
            print(f"Evaluation reward: {eval_reward:.2e}")
            print(f"Current epsilon: {self.epsilon:.3f}")
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                self.save("models/best_model.pt")
    
    def evaluate(self, env, n_episodes = 10):
        total_reward = 0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.act(obs, use_random=False)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            total_reward += episode_reward
        return total_reward / n_episodes
    
    def act(self, observation, use_random = False):
        if use_random or np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        observation = np.array(observation).reshape(1, -1)
        q_values = []
        
        for action in range(self.n_actions):
            if self.models[action] is not None:
                scaled_obs = self.scalers[action].transform(observation)
                q_values.append(self.models[action].predict(scaled_obs)[0])
            else:
                q_values.append(float('-inf'))
        
        return int(np.argmax(q_values))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'models': self.models,
            'scalers': self.scalers
        }
        joblib.dump(save_dict, path)

    def load(self):
        # Construct the correct path to the file
        path = os.path.join(os.path.dirname(__file__), "models", "best_model.pt")
        path = os.path.abspath(path)

        # Check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")
        
        # Load the file
        save_dict = joblib.load(path)
        self.models = save_dict['models']
        self.scalers = save_dict['scalers']

if __name__ == "__main__":
    agent = ProjectAgent()
    agent.train_fqi(n_epochs=1000, steps_per_epoch=100)
