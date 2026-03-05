"""
API file for generating reinforcement envs for population genetic simulators
"""
import importlib.util
import numpy as np
import msprime
from poprl.task import msprimeTask, SLiMTask
from poprl.envs.msprimeEnv import msprimeEnv
from poprl.envs.SLiMEnv import SLiMEnv

def simulate_target(model, mutation_rate):
    """Simulate 10 replicates of the target msprime model as a baseline for learning tasks"""
    if isinstance(model, str):
        spec = importlib.util.spec_from_file_location("target_model", model)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ts = mod.run()
        
        return ts.allele_frequency_spectrum()
    
    elif isinstance(model, tuple):
        demography, mutation_rate = model

    elif isinstance(model, msprime.Demography):
        demography = model
        if mutation_rate is None:
            mutation_rate = 1e-8
    
    else:
        raise ValueError("model must be an msprime.Demography or a (demography, mutation_rate) tuple")
    
    samples = {p.name: 10 for p in demography.populations}
    afs_list = []
        
    for _ in range(10):
        ts = msprime.sim_ancestry(samples=samples, demography=demography, sequence_length=1e6)
        ts = msprime.sim_mutations(ts, rate=mutation_rate)
        afs_list.append(ts.allele_frequency_spectrum())
            
    return np.mean(afs_list, axis=0)

def make_msprime(model, mutation_rate=None, task=None, tunable=None, randomize_start=False, max_steps=100, observation="sfs"):
    """Makes the msprime env compatible with gymnasium (see get_stdpopsim_model for getting demographic models)"""
    if task is None:
        task = msprimeTask(target=None, observation=observation)
    
    if task.target is None:
        task.target = simulate_target(model, mutation_rate)
    
    return msprimeEnv(model=model, task=task, tunable=tunable, randomize_start=randomize_start, max_steps=max_steps)

def make_slim(slim_file, mutation_rate=1e-7, observation="sfs", timeout=10.0):
    """Make the SLiM env compatible with gymnasium"""
    # Note: no simulation needed, as we use the burn in to set expectation
    task = SLiMTask(observation=observation, mutation_rate=mutation_rate)
    
    return SLiMEnv(slim_file=slim_file, task=task, timeout=timeout)
