"""
Registry of SFS task functions
"""
import warnings
import numpy as np
from gymnasium import spaces
from collections import deque

STACK_SIZE = 8
NUM_BINS = 21
NUM_BINS_MS = 100
DTYPE = np.float32
MAX_OBS_SIZE = 9261 # (21*3)

def compute_obs_size(samples):
    """Get size of flattened joint SFS from sample dict"""
    if samples is None:
        return NUM_BINS

    n_pops = len(samples)
    obs_size = int(np.prod([(n * 2 + 1) for n in samples.values()]))
    
    if obs_size > MAX_OBS_SIZE:
        warnings.warn(
            "Observation space will be large and convergence may be slow.",
            stacklevel=2,
        )

    return obs_size

def get_joint_sfs(afs_nd):
    """Flatten ND joint SFS from msprime"""
    afs = np.array(afs_nd, dtype=DTYPE)
    flat = afs.flatten()
    
    return np.where(flat > 0, flat, 1e-10).astype(DTYPE)

def get_sfs(afs, num_bins=NUM_BINS):
    """Bin AFS into fixed-size SFS, returns near-zero array if empty"""
    if afs is None or afs.sum() == 0:
        warnings.warn("Empty AFS; returning near-zero SFS.", stacklevel=2)
        return np.full(num_bins, 1e-10, dtype=DTYPE)
    
    afs = np.array(afs, dtype=DTYPE)
    afs = afs[1:-1]
    total = afs.sum()
    
    if total == 0:
        return np.full(num_bins, 1e-10, dtype=DTYPE)
    
    freqs = np.linspace(0, 1, len(afs))
    sfs = np.full(num_bins, 1e-10, dtype=DTYPE)
    bin_indices = np.floor(freqs * (num_bins - 1)).astype(int)
    
    for i, b in enumerate(bin_indices):
        sfs[b] += afs[i]
    
    sfs = np.where(sfs > 1e-10, sfs, 1e-10)
    
    return sfs.astype(DTYPE)

def get_sfs_from_ms(state_data, num_bins=NUM_BINS_MS):
    """Compute SFS from SLiM ms-format haplotype output"""
    lines = state_data.strip().split('\n')
    binary_lines = [line.strip() for line in lines if set(line.strip()).issubset({'0', '1'})]
    
    if not binary_lines:
        return np.full(num_bins, 1e-10, dtype=DTYPE)
    
    lengths = [len(line) for line in binary_lines]
    max_len = max(lengths)
    binary_lines = [line.ljust(max_len, '0')[:max_len] for line in binary_lines]
    data = np.array([[int(char) for char in line] for line in binary_lines])
    total = data.shape[0]
    column_sums = np.sum(data, axis=0)
    freqs = (column_sums / total) * 100.0
    buckets = np.floor(freqs).astype(int)
    valid_buckets = buckets[buckets < 100]
    sfs = np.full(num_bins, 1e-10, dtype=DTYPE)
    
    if valid_buckets.size > 0:
        unique_bins, counts = np.unique(valid_buckets, return_counts=True)
        
        for b, count in zip(unique_bins, counts):
            sfs[b] = count + 1e-10
    
    return sfs

def get_expectation_sfs(target, num_bins=NUM_BINS):
    """Compute target SFS from AFS or list of AFS"""
    if isinstance(target, np.ndarray):
        return [get_sfs(target, num_bins)] if target.ndim == 1 else [get_joint_sfs(target)]
    
    if isinstance(target, list):
        return [get_sfs(a, num_bins) if a.ndim == 1 else get_joint_sfs(a) for a in target]
    
    return None

def get_expectation_sfs_from_ms(state_data, num_bins=NUM_BINS_MS):
    """Compute target SFS from SLiM burn-in ms output"""
    ms_entries = state_data.strip().split('\n\n')
    
    return [get_sfs_from_ms(entry, num_bins) for entry in ms_entries]

def process_state(afs, sfs_stack, step_count=1, num_bins=NUM_BINS):
    """Update SFS stack from msprime AFS observation"""
    
    if step_count == 1 and sfs_stack.get('expectation') is None:
        warnings.warn("Expectation SFS not set at init; computing from step 1.", stacklevel=2)
        sfs_stack['expectation'] = get_expectation_sfs(afs, num_bins)

    if isinstance(afs, np.ndarray) and afs.ndim > 1:
        new_sfs = get_joint_sfs(afs)  
    
    else:
        new_sfs = get_sfs(afs, num_bins)
    
    sfs_stack['stack'].append(new_sfs)
    
    return np.stack(list(sfs_stack['stack']))

def process_state_from_ms(state_data, sfs_stack, step_count=1, num_bins=NUM_BINS_MS):
    """Update SFS stack from SLiM ms-format observation, set expectation at burn-in"""
    if step_count == 1 and sfs_stack.get('expectation') is None:
        sfs_stack['expectation'] = get_expectation_sfs_from_ms(state_data, num_bins)
    
    new_sfs = get_sfs_from_ms(state_data, num_bins)
    sfs_stack['stack'].append(new_sfs)
    
    return np.stack(list(sfs_stack['stack']))

def process_action(param_bounds=None):
    """Returns action processor that scales params by 0.9, 1.0, or 1.1"""
    action_map = {0: 0.9, 1: 1.0, 2: 1.1}
    
    def _process_action(action, current_params):
        new_params = {}
        
        for i, (key, value) in enumerate(current_params.items()):
            a = int(action[i]) if hasattr(action, '__len__') else int(action)
            multiplier = action_map[a]
            new_value = value * multiplier
            
            if param_bounds and key in param_bounds:
                lo, hi = param_bounds[key]
                new_value = np.clip(new_value, lo, hi)
            
            new_params[key] = new_value
        
        return new_params
    
    return _process_action

def process_action_slim(initial_rate=1e-7, param_bounds=None):
    """Returns action processor that adjusts mutation rate within SLiM episode"""
    action_map = {0: 0.9, 1: 1.0, 2: 1.1}
    state = [initial_rate]
    
    def _process_action(action):
        multiplier = action_map[int(action)]
        new_rate = np.clip(state[0] * multiplier, 1e-9, 1e-5)
        state[0] = new_rate
        return str(new_rate)
    
    return _process_action

def calculate_reward(next_state, expectation_sfs):
    """KL divergence between current and target SFS, negated as reward"""
    try:
        current_sfs = next_state[-1].astype(np.float32) + 1e-10
        klds = []
        
        for exp_sfs in expectation_sfs:
            exp_sfs = np.array(exp_sfs, dtype=DTYPE) + 1e-10
            current_normalized = current_sfs / np.sum(current_sfs)
            expectation_normalized = exp_sfs / np.sum(exp_sfs)
            kld = np.sum(expectation_normalized * (np.log(expectation_normalized) - np.log(current_normalized)))
            
            if np.isfinite(kld):
                klds.append(kld)
        
        if not klds:
            return -100000.0
        
        return float(-np.mean(klds))
   
    except Exception:
        return -10000.0

def get_action_space(n_params=1):
    """MultiDiscrete action space with 3 choices per parameter"""
    return spaces.MultiDiscrete([3] * n_params)

def get_action_space_slim():
    """Discrete action space with 3 choices for mutation rate control"""
    return spaces.Discrete(3)

def get_observation_space(sfs_stack_size=STACK_SIZE, num_bins=NUM_BINS):
    """Stacked SFS observations for msprime env"""
    return spaces.Box(low=0, high=np.inf, shape=(sfs_stack_size, obs_size), dtype=DTYPE)

def get_observation_space_slim(sfs_stack_size=STACK_SIZE, num_bins=NUM_BINS_MS):
    """Stacked SFS observations for SLiM env"""
    return spaces.Box(low=0, high=np.inf, shape=(sfs_stack_size, num_bins), dtype=DTYPE)

def get_initial_state(target=None, sfs_stack_size=STACK_SIZE, obs_size=NUM_BINS):
    """Initialize zeroed SFS stack and context with target expectation"""
    stack = deque(maxlen=sfs_stack_size)
    
    for _ in range(sfs_stack_size):
        stack.append(np.full(obs_size, 1e-10, dtype=DTYPE))

    if target is not None
        expectation = get_expectation_sfs(target)
    
    else: 
        expectation = None
    
    sfs_stack = {'stack': stack, 'expectation': expectation}
    
    return np.stack(list(stack)), sfs_stack

def get_initial_state_slim(context, sfs_stack_size=STACK_SIZE, num_bins=NUM_BINS_MS):
    """Initialize zeroed SFS stack in SLiM context, expectation set at burn-in"""
    if 'stack' not in context:
        context['stack'] = deque(maxlen=sfs_stack_size)
    
    if 'expectation' not in context:
        context['expectation'] = None
    
    context['stack'].clear()
    
    for _ in range(sfs_stack_size):
        context['stack'].append(np.full(num_bins, 1e-10, dtype=DTYPE))
    
    return np.stack(list(context['stack']))

def calculate_reward_from_context(next_state, context):
    """Wrapper to pull expectation from context and compute KL reward"""
    expectation = context.get('expectation')
    
    if expectation is None:
        return 0.0
    
    return calculate_reward(next_state, expectation)




