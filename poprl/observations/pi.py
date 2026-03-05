"""
Registry of nucleotide diversity task functions
"""
import warnings
import numpy as np
from gymnasium import spaces
from collections import deque

STACK_SIZE = 8
DTYPE = np.float32

def compute_pi_from_afs(afs):
    """Compute nucleotide diversity from an allele frequency spectrum"""
    afs = np.array(afs, dtype=DTYPE)
    afs = afs[1:-1]
    n = len(afs) + 1
    freqs = np.arange(1, n) / n
    pi = float(np.sum(2 * freqs * (1 - freqs) * afs))
    
    return pi if np.isfinite(pi) else 0.0

def compute_pi_from_ms(state_data):
    """Compute nucleotide diversity from ms-format haplotype output"""
    lines = state_data.strip().split('\n')
    binary_lines = [line.strip() for line in lines if set(line.strip()).issubset({'0', '1'}) and len(line.strip()) > 0]
    
    if not binary_lines:
        return 0.0
    
    lengths = [len(line) for line in binary_lines]
    max_len = max(lengths)
    binary_lines = [line.ljust(max_len, '0')[:max_len] for line in binary_lines]
    data = np.array([[int(c) for c in line] for line in binary_lines], dtype=DTYPE)
    n = data.shape[0]
    
    if n < 2:
        return 0.0
    
    freqs = data.mean(axis=0)
    pi = float(np.sum(2 * freqs * (1 - freqs)))
    
    return pi if np.isfinite(pi) else 0.0

def get_expectation_pi(target):
    """Compute expected pi from target AFS or list of AFS"""
    if isinstance(target, np.ndarray):
        return [compute_pi_from_afs(target)]
    
    if isinstance(target, list):
        return [compute_pi_from_afs(a) for a in target]
    
    return None

def process_state(afs, context, step_count=1):
    """Update pi stack from msprime AFS observation"""
    if 'stack' not in context:
        context['stack'] = deque(maxlen=STACK_SIZE)
        
        for _ in range(STACK_SIZE):
            context['stack'].append(np.array([0.0], dtype=DTYPE))
    
    if step_count == 1 and context.get('expectation') is None:
        warnings.warn("Expectation pi not set at init; computing from step 1.", stacklevel=2)
        context['expectation'] = get_expectation_pi(afs)
    
    pi = compute_pi_from_afs(afs)
    context['stack'].append(np.array([pi], dtype=DTYPE))
    
    return np.stack(list(context['stack']))

def process_state_from_ms(state_data, context, step_count=1):
    """Update pi stack from SLiM ms-format observation"""
    if 'stack' not in context:
        context['stack'] = deque(maxlen=STACK_SIZE)
        
        for _ in range(STACK_SIZE):
            context['stack'].append(np.array([0.0], dtype=DTYPE))
    
    if 'expectation' not in context:
        context['expectation'] = None
    
    if step_count == 1 and context.get('expectation') is None:
        # Use first generation as burn-in expectation
        ms_entries = state_data.strip().split('\n\n')
        context['expectation'] = [compute_pi_from_ms(entry) for entry in ms_entries]
    
    pi = compute_pi_from_ms(state_data)
    context['stack'].append(np.array([pi], dtype=DTYPE))
    
    return np.stack(list(context['stack']))

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

def process_action_slim(initial_rate=1e-7):
    """Returns action processor that adjusts mutation rate within SLiM episode"""
    action_map = {0: 0.9, 1: 1.0, 2: 1.1}
    state = [initial_rate]
    
    def _process_action(action):
        multiplier = action_map[int(action)]
        new_rate = np.clip(state[0] * multiplier, 1e-9, 1e-5)
        state[0] = new_rate
        return str(new_rate)
    
    return _process_action

def calculate_reward_from_context(next_state, context):
    """Negative MSE between current pi and target expectation"""
    expectation = context.get('expectation')
    
    if expectation is None:
        return 0.0
    
    try:
        current_pi = float(next_state[-1][0])
        errors = [(current_pi - exp_pi) ** 2 for exp_pi in expectation if np.isfinite(exp_pi)]
        
        if not errors:
            return 0.0
        
        return float(-np.mean(errors))
    
    except Exception:
        return -10000.0

def get_action_space(n_params=1):
    """MultiDiscrete action space with 3 choices per parameter"""
    return spaces.MultiDiscrete([3] * n_params)

def get_action_space_slim():
    """Discrete action space with 3 choices for mutation rate control"""
    return spaces.Discrete(3)

def get_observation_space(sfs_stack_size=STACK_SIZE):
    """Stacked pi observations for msprime env"""
    return spaces.Box(low=0.0, high=np.inf, shape=(sfs_stack_size, 1), dtype=DTYPE)

def get_observation_space_slim(sfs_stack_size=STACK_SIZE):
    """Stacked pi observations for SLiM env"""
    return spaces.Box(low=0.0, high=np.inf, shape=(sfs_stack_size, 1), dtype=DTYPE)

def get_initial_state(target=None, sfs_stack_size=STACK_SIZE):
    """Initialize zeroed pi stack and context with target expectation"""
    stack = deque(maxlen=sfs_stack_size)
    
    for _ in range(sfs_stack_size):
        stack.append(np.array([0.0], dtype=DTYPE))
    
    expectation = get_expectation_pi(target) if target is not None else None
    context = {'stack': stack, 'expectation': expectation}
    
    return np.stack(list(stack)), context

def get_initial_state_slim(context, sfs_stack_size=STACK_SIZE):
    """Initialize zeroed pi stack in SLiM context, expectation set at burn-in"""
    context['stack'] = deque(maxlen=sfs_stack_size)
    context['expectation'] = None
    
    for _ in range(sfs_stack_size):
        context['stack'].append(np.array([0.0], dtype=DTYPE))
    
    return np.stack(list(context['stack']))


