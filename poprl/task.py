""" 
Task management file for the PopRL package. SLiM tasks attempt to maintain the
statistic throughout a simulation. msprime tasks attempt to match the target statistic
set by the user. Current implementation is for the SFS, and for nucleotide diversity (pi)
"""
from poprl.observations import sfs
from poprl.observations import pi

# Sets env functions based on backend and stat used
MSPRIME_OBSERVATION_REGISTRY = {
    "sfs": {
        "process_state": sfs.process_state,
        "process_action": sfs.process_action(),
        "calculate_reward": sfs.calculate_reward_from_context,
        "action_space": sfs.get_action_space,
        "observation_space": sfs.get_observation_space,
        "get_initial_state": sfs.get_initial_state,
    },
    
    "pi": {
        "process_state": pi.process_state,
        "process_action": pi.process_action(),
        "calculate_reward": pi.calculate_reward_from_context,
        "action_space": pi.get_action_space,
        "observation_space": pi.get_observation_space,
        "get_initial_state": pi.get_initial_state,
    },
}

SLIM_OBSERVATION_REGISTRY = {
    "sfs": {
        "process_state": sfs.process_state_from_ms,
        "process_action": sfs.process_action_slim,
        "calculate_reward": sfs.calculate_reward_from_context,
        "action_space": sfs.get_action_space_slim,
        "observation_space": sfs.get_observation_space_slim,
        "get_initial_state": sfs.get_initial_state_slim,
    },
    
    "pi": {
        "process_state": pi.process_state_from_ms,
        "process_action": pi.process_action_slim,
        "calculate_reward": pi.calculate_reward_from_context,
        "action_space": pi.get_action_space_slim,
        "observation_space": pi.get_observation_space_slim,
        "get_initial_state": pi.get_initial_state_slim,
    },
}

# Defines env task based on backend
class msprimeTask:
    def __init__(self, target, observation="sfs", parameters=None):
        if observation not in MSPRIME_OBSERVATION_REGISTRY:
            raise ValueError(f"Unknown observation type '{observation}'. Available: {list(MSPRIME_OBSERVATION_REGISTRY)}")
        
        self.target = target
        self.observation = observation
        self.parameters = parameters
        obs = MSPRIME_OBSERVATION_REGISTRY[observation]
        self.process_state = obs["process_state"]
        self.process_action = obs["process_action"]
        self.calculate_reward = obs["calculate_reward"]
        self.action_space = obs["action_space"]
        self.observation_space = obs["observation_space"]()
        self.get_initial_state = obs["get_initial_state"]

class SLiMTask:
    def __init__(self, observation="sfs", mutation_rate=1e-7):
        if observation not in SLIM_OBSERVATION_REGISTRY:
            raise ValueError(f"Unknown observation type '{observation}'. Available: {list(SLIM_OBSERVATION_REGISTRY)}")
        
        self.observation = observation
        self.mutation_rate = mutation_rate
        obs = SLIM_OBSERVATION_REGISTRY[observation]
        self.process_state = obs["process_state"]
        self.process_action = obs["process_action"](initial_rate=mutation_rate)
        self.calculate_reward = obs["calculate_reward"]
        self.action_space = obs["action_space"]()
        self.observation_space = obs["observation_space"]()
        self.get_initial_state = obs["get_initial_state"]
