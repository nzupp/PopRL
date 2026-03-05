# PopRL
PopRL is a Gymnasium wrapper for the msprime and SLiM 5 simulators, enabling RL-based exploration of evolutionary dynamics and demographic inference.

## General overview
Reinforcement learning (RL) is a machine learning approach where agents learn optimal actions through interaction with an environment, while Gymnasium (formerly OpenAI Gym) provides standardized environments for developing and comparing RL algorithms. SLiM is a powerful forward-time simulation software that models evolutionary processes by tracking individuals and their genomes across generations. msprime is a coalescent based backward-time simulator that efficiently models large population histories, enabling inference of demographic parameters from genetic data. PopRL unifies these simulators under a single Gymnasium interface, allowing researchers to apply RL to both individual level forward-time dynamics and large scale demographic inference.

## Requirements
- Python >=3.8
- SLiM 5.0 or 5.1
- msprime >=1.4.0
- Stable-Baselines3 for reinforcement learning example

## Quick start guide
1. Clone repository and `cd` into directory
2. Install via pip: `pip install -e .`
3. Install SLiM 5 from the [Messer Lab](https://messerlab.org/slim/) and ensure it's in your system PATH or working directory
4. Run tests with `pytest -v`

## Components
### SLiM evolutionary model
PopRL SLiM integration utilizes Eidos scripts created for running evolutionary simulations with SLiM. The SLiM 5 documentation goes into extensive detail regarding the specifics of these tools, but in short SLiM is highly powerful and flexible, able to model most evolutionary scenarios. The PopRL package includes three established SLiM scripts- one tracking a single population experiencing a bottleneck, and another that tracks population growth, and finally an island model of a population with fluctuating size.

All of these PopRL models contain 'hooks', or additions to the underlying model, that enable SLiM to communicate with the rest of the PopRL framework. For a SLiM script to be compatible with PopRL, it needs to implement several specific hooks:

1. **Flag File Definition**:
```code
defineConstant("FLAG_FILE", "flag.txt");
```
This defines a flag file that serves as a communication channel between SLiM and the Python environment.

2. **State Output**:
```code
g = p1.sampleIndividuals(100).genomes;
g.outputMS("state.txt", append=T);
```
This hook outputs the current state of the simulation in MS format, which the environment uses to construct observations.

3. **Action Reading**:
```code
while (fileExists(FLAG_FILE) == F) {}
mutRateStr = readFile(FLAG_FILE);
while (size(mutRateStr) == 0) {
    mutRateStr = readFile(FLAG_FILE);
}
mutRate = asFloat(mutRateStr);
sim.chromosome.setMutationRate(mutRate);
```
This hook waits for the flag file to exist, reads the action value from it (in this case, a new mutation rate), and applies the action to the simulation.

4. **Simulation Completion Signal**:
```code
writeFile("generation_complete.txt", "1");
```
This hook signals that the simulation has completed, allowing the Python environment to clean up resources.

5. **File Cleanup**:
```code
deleteFile(FLAG_FILE);
```
This hook deletes the flag file after reading the action, preparing for the next communication cycle.

### Tasks
PopRL defines tasks as the combination of an observation type, reward function and action processing logic. Two tasks classes are provided: `msprimeTask` for coalescent based inference and `SLiMTask` for forward-time simulation. Both accept an observation argument (`sfs` or `pi`) that determines how raw simulator output is processed into agent observations and how rewards are computed. `msprimeTask` additionally accepts a target, either a user supplied allele frequency spectra, or one simulated from stdpopsim demographic models, against which summary statistics are evaluated.

### Environments
`msprimeEnv` and `SLiMEnv` are the Gymnasium wrappers that accept a task and a simulator backend and handle the episode loop, state management, and communication with the underlying simulator. Because observation type, reward function and action processing are fully defined by the task, swapping between `sfs` and `pi` or adding new statistics requires no changes to the environment code. Custom observations can be added by extending the observation registry with new processing and reward functions.

## Core API
PopRL provides two factory functions for creating Gymnasium-compatible training environments: `make_msprime` for coalescent based demographic inference and `make_slim` for forward-time simulation. Both accept an observation type argument to select between `sfs` and `pi`, and return environments compatible with standard RL libraries like Stable-Baselines3.

```python
from poprl import make_msprime
from catalog import get_model

# Load a published demographic model from stdpopsim
model = get_model("HomSap", "Africa_1T12")

# Create environment with SFS or pi observation
env = make_msprime(
    model,
    tunable=["pop_AFR_initial_size", "event_0_initial_size"],  # parameters the agent can adjust
    randomize_start=True,  # initialize agent at randomized parameter estimates
    max_steps=100,
    observation="sfs",  # or "pi"
)
```

```python
from poprl import make_slim

# Use a built-in SLiM model or provide a path to a custom script
env = make_slim(
    "examples/bottleneck.slim",  # or "examples/growth.slim", "examples/island.slim"
    mutation_rate=1e-7,
    observation="sfs",  # or "pi"
)
```

## Creating Custom Environments

`PopRL` allows for the creation of custom tasks tailored to specific evolutionary questions. Custom observations can be added by implementing six functions and registering them in the observation registry in `poprl/tasks.py`:

```python
MSPRIME_OBSERVATION_REGISTRY["my_stat"] = {
    "process_state": my_stat.process_state,
    "process_action": my_stat.process_action(),
    "calculate_reward": my_stat.calculate_reward_from_context,
    "action_space": my_stat.get_action_space,
    "observation_space": my_stat.get_observation_space,
    "get_initial_state": my_stat.get_initial_state,
}
```

Once registered, `msprimeTask` and `SLiMTask` will support the new observation type without any changes to the environment code. See `poprl/observations/sfs.py` and `poprl/observations/pi.py` for reference implementations.

## Worked example
PopRL includes three built-in `SLiM` models (bottleneck, growth, and island) and example scripts for both `SLiM` and `msprime` training. The following demonstrates training a PPO agent on a published human demographic model using the `msprime` environment.

The task is demographic inference: the agent observes the SFS produced under current parameter estimates and adjusts population sizes and event times to match a target allele frequency spectrum simulated from a `stdpopsim` model. The agent has three actions per tunable parameter: decrease by 10%, hold, or increase by 10%. Reward is computed as negative KL divergence between the observed and target SFS.

Import the required packages.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from poprl import make_msprime
from catalog import get_model
```

Load a published human demographic model from stdpopsim. To print other available models, use `catalog.avail_stdpopsim()`.

```python
model = get_model("HomSap", "Africa_1T12")
```

Create the env, using `make_msprime`.

```python
env = make_msprime(
    model,
    tunable=["pop_AFR_initial_size", "event_0_initial_size"],
    randomize_start=True,
    max_steps=100,
    observation="sfs",
)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)
```

Train the agent, using the newly created env. Here we use Stable Baselines3's PPO model.

```python
agent = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64)
agent.learn(total_timesteps=10000)
agent.save("ppo_africa_1t12_sfs")
```

See examples/ for a complete `SLiM` training run across all three built-in models, and for a pi-based version of the `msprime` example.

## Troubleshooting
If you encounter any issues with installation or simulation errors, please report them on our GitHub issue tracker [link]. Common issues include SLiM not being found in PATH (ensure `SLiM` is properly installed and accessible from your command line) and simulation failures due to biologically implausible parameter settings.
