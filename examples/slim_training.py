"""
SLiM training example task
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from poprl import make_slim

# available built in SLiM models
SLIM_MODELS = [
    ("examples/bottleneck.slim", "bottleneck"),
    ("examples/growth.slim",     "growth"),
    ("examples/island.slim",     "island"),
]

MUTATION_RATE = 1e-7
TOTAL_TIMESTEPS = 50000
results = {}

for slim_file, name in SLIM_MODELS:
    env = make_slim(slim_file, mutation_rate=MUTATION_RATE)
    env = DummyVecEnv([lambda e=env: e])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    agent = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
    )
    agent.learn(total_timesteps=TOTAL_TIMESTEPS)

    # Optionally save results
    # agent.save(f"ppo_{name}")
    # env.save(f"vecnormalize_{name}.pkl")
    # print(f"Model saved to ppo_{name}.zip")
    env.close()

print("\nAll SLiM training complete.")