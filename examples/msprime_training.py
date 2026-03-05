"""
msprime example using stdpopsim demography, for both SFS and Pi tasks
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from poprl import make_msprime
from catalog import get_model

# Human model from stdpopsim
model = get_model("HomSap", "Africa_1T12")

# Create env for SFS task, set tunable params if desired
env_sfs = make_msprime(
    model,
    tunable=["pop_AFR_initial_size", "event_0_initial_size"],
    randomize_start=True,
    max_steps=100,
    observation="sfs",
)

env_sfs = DummyVecEnv([lambda: env_sfs])
env_sfs = VecNormalize(env_sfs, norm_obs=True, norm_reward=True)
agent_sfs = PPO("MlpPolicy", env_sfs, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2)
agent_sfs.learn(total_timesteps=100)

# Optionally save results:
# agent_sfs.save("ppo_africa_1t12_sfs")
# env_sfs.save("vecnormalize_africa_1t12_sfs.pkl")

# Create env for Pi task
env_pi = make_msprime(
    model,
    tunable=["pop_AFR_initial_size", "event_0_initial_size"],
    randomize_start=True,
    max_steps=100,
    observation="pi",
)

env_pi = DummyVecEnv([lambda: env_pi])
env_pi = VecNormalize(env_pi, norm_obs=True, norm_reward=True)
agent_pi = PPO("MlpPolicy", env_pi, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2)
agent_pi.learn(total_timesteps=100)

# Optionally save results:
# agent_pi.save("ppo_africa_1t12_pi")
# env_pi.save("vecnormalize_africa_1t12_pi.pkl")