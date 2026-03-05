"""Unit tests for MsprimeTask and SlimTask configuration and handling"""
import pytest
from poprl.task import msprimeTask, SLiMTask

def test_msprime_task_sfs():
    """Test if MsprimeTask initializes correctly with SFS observation"""
    task = MsprimeTask(target=None, observation="sfs")
    assert task.observation == "sfs"
    assert callable(task.process_state)
    assert callable(task.process_action)
    assert callable(task.calculate_reward)

def test_msprime_task_pi():
    """Test if MsprimeTask initializes correctly with pi observation"""
    task = MsprimeTask(target=None, observation="pi")
    assert task.observation == "pi"

def test_msprime_task_invalid():
    """Test if MsprimeTask raises ValueError for invalid observation"""
    with pytest.raises(ValueError):
        MsprimeTask(target=None, observation="invalid")

def test_slim_task_sfs():
    """Test if SlimTask initializes correctly with SFS observation"""
    task = SlimTask(observation="sfs", mutation_rate=1e-7)
    assert task.observation == "sfs"
    assert callable(task.process_state)
    assert callable(task.calculate_reward)

def test_slim_task_pi():
    """Test if SlimTask initializes correctly with pi observation"""
    task = SlimTask(observation="pi", mutation_rate=1e-7)
    assert task.observation == "pi"

def test_slim_task_invalid():
    """Test if SlimTask raises ValueError for invalid observation"""
    with pytest.raises(ValueError):
        SlimTask(observation="invalid")

def test_msprime_task_observation_space_shape():
    """Test if MsprimeTask SFS observation space has correct shape"""
    from poprl.observations import sfs
    task = MsprimeTask(target=None, observation="sfs")
    assert task.observation_space.shape == (sfs.STACK_SIZE, sfs.NUM_BINS)

def test_msprime_task_pi_observation_space_shape():
    """Test if MsprimeTask pi observation space has correct shape"""
    from poprl.observations import pi
    task = MsprimeTask(target=None, observation="pi")
    assert task.observation_space.shape == (pi.STACK_SIZE, 1)
