"""Unit tests for env reset, step, reward, and termination behavior"""
import numpy as np
import pytest
import shutil
from poprl import make_msprime, make_slim
from poprl.catalog import get_model

slim_available = pytest.mark.skipif(
    shutil.which("slim") is None,
    reason="SLiM not installed"
)

@pytest.fixture
def env():
    """Test if standard environment fixture initializes correctly"""
    model = get_model("HomSap", "Africa_1T12")
    env = make_msprime(model, tunable=["event_0_initial_size"], randomize_start=False, max_steps=5)
    yield env
    env.close()

@pytest.fixture
def slim_env():
    """Test if SLiM environment fixture initializes correctly"""
    env = make_slim("bottleneck.slim")
    yield env
    env.close()

def test_reset_returns_correct_shape(env):
    """Test if reset returns observation with correct shape"""
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape

def test_step_returns_correct_shape(env):
    """Test if step returns observation with correct shape"""
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == env.observation_space.shape

def test_reward_is_finite(env):
    """Test if reward from step is finite"""
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert np.isfinite(reward)

def test_terminates_at_max_steps(env):
    """Test if environment terminates at max_steps"""
    env.reset()
    terminated = False
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    assert terminated

def test_params_change_on_action(env):
    """Test if parameters change when non hold action applied"""
    env.reset()
    params_before = dict(env.current_params)
    env.step(np.array([2] * len(env.current_params)))
    assert env.current_params != params_before

def test_params_unchanged_on_hold(env):
    """Test if parameters remain unchanged on hold action"""
    env.reset()
    params_before = dict(env.current_params)
    env.step(np.array([1] * len(env.current_params)))
    assert env.current_params == params_before

def test_randomize_start_differs_from_true():
    """Test if randomize_start changes initial parameters"""
    model = get_model("HomSap", "Africa_1T12")
    env_rand = make_msprime(model, tunable=["event_0_initial_size"], randomize_start=True, max_steps=5)
    env_fixed = make_msprime(model, tunable=["event_0_initial_size"], randomize_start=False, max_steps=5)
    env_rand.reset()
    env_fixed.reset()
    assert env_rand.current_params != env_fixed.current_params
    env_rand.close()
    env_fixed.close()

def test_pi_observation_env():
    """Test if environment supports pi observation"""
    model = get_model("HomSap", "Africa_1T12")
    env = make_msprime(model, tunable=["event_0_initial_size"], max_steps=5, observation="pi")
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    env.close()

@slim_available
def test_slim_env_reset_returns_correct_shape(slim_env):
    """Test if SLiM environment reset returns correct shape"""
    obs, info = slim_env.reset()
    assert obs.shape == slim_env.observation_space.shape

@slim_available
def test_slim_env_step_returns_correct_shape(slim_env):
    """Test if SLiM environment step returns correct shape"""
    slim_env.reset()
    action = slim_env.action_space.sample()
    obs, reward, terminated, truncated, info = slim_env.step(action)
    assert obs.shape == slim_env.observation_space.shape

@slim_available
def test_slim_env_reward_is_finite(slim_env):
    """Test if SLiM environment reward is finite"""
    slim_env.reset()
    action = slim_env.action_space.sample()
    obs, reward, terminated, truncated, info = slim_env.step(action)
    assert np.isfinite(reward)

@slim_available
def test_slim_env_terminates_at_max_steps(slim_env):
    """Test if SLiM environment terminates at max_steps"""
    slim_env.reset()
    terminated = False
    for _ in range(5):
        action = slim_env.action_space.sample()
        obs, reward, terminated, truncated, info = slim_env.step(action)
    assert terminated

@slim_available
def test_slim_env_pi_observation():
    """Test if SLiM environment supports pi observation"""
    env = make_slim("bottleneck.slim", observation="pi")
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    env.close()
