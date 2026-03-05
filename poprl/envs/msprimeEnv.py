"""
msprime gym env
"""
import msprime
import numpy as np
import gymnasium as gym
import importlib.util
import logging

logger = logging.getLogger(__name__)

class MSPRIMEEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, model, task, tunable=None, randomize_start=False, max_steps=100):
        super().__init__()
        self.model = model
        self.task = task
        self.step_count = 0
        self.previous_state = None
        self.context = None
        self.tunable = set(tunable) if tunable is not None else set()
        self.randomize_start = randomize_start
        self.max_steps = max_steps

        if isinstance(model, str):
            spec = importlib.util.spec_from_file_location("msprime_model", model)
            self.mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.mod)
            self.demography = None
            self.mutation_rate = None
            self.samples = None
            self.current_params = dict(task.parameters) if task.parameters else {}
            self._tunable_events = []
            logger.info(f"msprime model loaded from file: {model}")
        
        else:
            self.mod = None
            self.demography, self.mutation_rate = model
            self.samples = {p.name: 10 for p in self.demography.populations}
            self._tunable_events = [
                i for i, e in enumerate(self.demography.events)
                if type(e).__name__ == 'PopulationParametersChange'
                and e.initial_size is not None
            ]
            
            if not self.tunable:
                self.tunable = {f'event_{i}_initial_size' for i in self._tunable_events} | \
                               {f'event_{i}_time' for i in self._tunable_events}
            self.current_params = self._build_params(randomize=self.randomize_start)
            logger.info(f"msprime model loaded, tunable params: {self.tunable}")

        self.action_space = task.action_space(len(self.current_params))
        self.observation_space = task.observation_space

    def _build_params(self, randomize=False):
        """Build parameter dict from demography, optionally randomizing within 0.5x-2x of true values"""
        params = {}
        
        for i in self._tunable_events:
            e = self.demography.events[i]
            
            if f'event_{i}_initial_size' in self.tunable:
                v = e.initial_size
                params[f'event_{i}_initial_size'] = np.random.uniform(v / 2, v * 2) if randomize else v
            
            if f'event_{i}_time' in self.tunable:
                v = e.time
                params[f'event_{i}_time'] = np.random.uniform(v / 2, v * 2) if randomize else v
        
        for p in self.demography.populations:
            
            if f'pop_{p.name}_initial_size' in self.tunable:
                v = p.initial_size
                params[f'pop_{p.name}_initial_size'] = np.random.uniform(v / 2, v * 2) if randomize else v
        
        return params

    def _rebuild_demography(self):
        """Reconstruct msprime demography object from current parameter estimates"""
        d = msprime.Demography()
        
        for p in self.demography.populations:
            initial_size = self.current_params.get(f'pop_{p.name}_initial_size', p.initial_size)
            d.add_population(
                name=p.name,
                initial_size=initial_size,
                growth_rate=p.growth_rate,
            )
        
        for i, row in enumerate(self.demography.migration_matrix):
            
            for j, rate in enumerate(row):
                
                if i != j and rate > 0:
                    d.set_migration_rate(i, j, rate)
        
        for i, e in enumerate(self.demography.events):
            
            if i in self._tunable_events:
                pop = None if e.population == -1 else e.population
                d.add_population_parameters_change(
                    time=self.current_params.get(f'event_{i}_time', e.time),
                    initial_size=self.current_params.get(f'event_{i}_initial_size', e.initial_size),
                    growth_rate=e.growth_rate if e.growth_rate is not None else 0,
                    population=pop,
                )
            
            else:
                d.events.append(e)
        
        d.sort_events()
        return d

    def _run_simulation(self):
        """Run msprime simulation and return allele frequency spectrum"""
        if self.mod is not None:
            ts = self.mod.run(**self.current_params)
        
        else:
            demography = self._rebuild_demography()
            ts = msprime.sim_ancestry(samples=self.samples, demography=demography, sequence_length=1e6)
            ts = msprime.sim_mutations(ts, rate=self.mutation_rate)
        
        return ts.allele_frequency_spectrum()

    def step(self, action):
        """Apply action, run simulation, return observation and reward"""
        self.step_count += 1
        self.current_params = self.task.process_action(action, self.current_params)
        
        try:
            state_data = self._run_simulation()
        
        except Exception as e:
            logger.error(f"Simulation failed at step {self.step_count}: {e}")
            return self.previous_state, -1, True, False, {"error": str(e)}
        
        current_state = self.task.process_state(state_data, self.context, self.step_count)
        reward = 0
        
        if self.previous_state is not None:
            reward = self.task.calculate_reward(current_state, self.context)
        
        self.previous_state = current_state
        terminated = self.step_count >= self.max_steps
        logger.info(f"Step {self.step_count}, reward: {reward:.4f}")
        
        return current_state, reward, terminated, False, {}

    def reset(self, seed=None):
        """Reset env and resample starting params if randomize_start is set"""
        super().reset(seed=seed)
        self.step_count = 0
        self.previous_state = None
        
        if self.demography is not None:
            self.current_params = self._build_params(randomize=self.randomize_start)
        
        else:
            self.current_params = dict(self.task.parameters) if self.task.parameters else {}
        
        initial_state, self.context = self.task.get_initial_state(self.task.target)
        self.previous_state = initial_state
        logger.info("MSPRIMEEnv reset")
        
        return initial_state, {}

    def close(self):
        pass
