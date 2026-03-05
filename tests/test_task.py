"""Unit tests for MsprimeTask and SlimTask configuration and handling"""
import pytest
from poprl.task import msprimeTask, SLiMTask

def test_msprime_task_sfs():
    """Test if msprimeTask initializes correctly with SFS observation"""
    task = msprimeTask(target=None, observation="sfs")
    assert task.observation == "sfs"
    assert callable(task.process_state)
    assert callable(task.process_action)
    assert callable(task.calculate_reward)

def test_msprime_task_pi():
    """Test if msprimeTask initializes correctly with pi observation"""
    task = msprimeTask(target=None, observation="pi")
    assert task.observation == "pi"

def test_msprime_task_invalid():
    """Test if msprimeTask raises ValueError for invalid observation"""
    with pytest.raises(ValueError):
        msprimeTask(target=None, observation="invalid")

def test_slim_task_sfs():
    """Test if SLiMTask initializes correctly with SFS observation"""
    task = SLiMTask(observation="sfs", mutation_rate=1e-7)
    assert task.observation == "sfs"
    assert callable(task.process_state)
    assert callable(task.calculate_reward)

def test_slim_task_pi():
    """Test if SLiMTask initializes correctly with pi observation"""
    task = SLiMTask(observation="pi", mutation_rate=1e-7)
    assert task.observation == "pi"

def test_slim_task_invalid():
    """Test if SLiMTask raises ValueError for invalid observation"""
    with pytest.raises(ValueError):
        SLiMTask(observation="invalid")

def test_msprime_task_observation_space_shape():
    """Test if msprimeTask SFS observation space has correct shape"""
    from poprl.observations import sfs
    task = msprimeTask(target=None, observation="sfs")
    assert task.observation_space.shape == (sfs.STACK_SIZE, sfs.NUM_BINS)

def test_msprime_task_pi_observation_space_shape():
    """Test if msprimeTask pi observation space has correct shape"""
    from poprl.observations import pi
    task = msprimeTask(target=None, observation="pi")
    assert task.observation_space.shape == (pi.STACK_SIZE, 1)
