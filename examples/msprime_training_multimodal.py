"""
Iterate over multiple demographic models and train a separate agent for each
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from poprl import make_msprime
from poprl.catalog import get_model

MODELS = [
    ("HomSap", "Africa_1T12",       ["pop_AFR_initial_size", "event_0_initial_size"]),
    ("HomSap", "African2Epoch_1H18", ["event_0_initial_size"]),
    ("HomSap", "Africa_1B08",        ["event_0_initial_size"]),
]

TOTAL_TIMESTEPS = 100000
MAX_STEPS = 100
results = {}

for species_id, model_id, tunable in MODELS:
    print(f"\n{'='*60}")
    print(f"Training on {species_id} / {model_id}")
    print(f"Tunable params: {tunable}")
    print(f"{'='*60}")

    model = get_model(species_id, model_id)
    if model is None:
        print(f"Skipping {model_id} - model not found")
        continue

    env = make_msprime(model, tunable=tunable, randomize_start=True, max_steps=MAX_STEPS)
    env = DummyVecEnv([lambda e=env: e])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    agent = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4)
    agent.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Evaluate: run 10 episodes and record mean reward
    episode_rewards = []
    for _ in range(10):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward[0]
        episode_rewards.append(float(episode_reward))

    mean_reward = sum(episode_rewards) / len(episode_rewards)
    results[f"{species_id}/{model_id}"] = {
        "tunable": tunable,
        "mean_eval_reward": mean_reward,
        "n_eval_episodes": len(episode_rewards)
    }
    print(f"Mean eval reward: {mean_reward:.4f}")
    env.close()

# Optionally save results
# with open("multimodel_results.json", "w") as f:
#     json.dump(results, f, indent=2)

# print("\nAll results saved to multimodel_results.json")