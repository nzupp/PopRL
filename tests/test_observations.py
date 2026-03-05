"""Unit tests for SFS and pi observation and reward utilities"""
import numpy as np
import pytest
from poprl.observations import sfs, pi

def test_get_sfs_shape():
    """Test if SFS output has correct bin shape"""
    afs = np.zeros(22)
    afs[1:5] = [10, 5, 3, 1]
    result = sfs.get_sfs(afs)
    assert result.shape == (sfs.NUM_BINS,)

def test_get_sfs_empty():
    """Test if empty AFS yields positive smoothed SFS"""
    afs = np.zeros(22)
    result = sfs.get_sfs(afs)
    assert result.shape == (sfs.NUM_BINS,)
    assert np.all(result > 0)

def test_get_sfs_normalized():
    """Test if nonzero AFS produces strictly positive SFS"""
    afs = np.zeros(22)
    afs[1:5] = [10, 5, 3, 1]
    result = sfs.get_sfs(afs)
    assert np.all(result > 0)

def test_process_state_sfs_shape():
    """Test if processed SFS state has expected stack shape"""
    afs = np.zeros(22)
    afs[1:5] = [10, 5, 3, 1]
    obs, ctx = sfs.get_initial_state(target=afs)
    result = sfs.process_state(afs, ctx, step_count=2)
    assert result.shape == (sfs.STACK_SIZE, sfs.NUM_BINS)

def test_calculate_reward_sfs_finite():
    """Test if SFS reward computation returns finite value"""
    afs = np.zeros(22)
    afs[1:5] = [10, 5, 3, 1]
    obs, ctx = sfs.get_initial_state(target=afs)
    result = sfs.process_state(afs, ctx, step_count=2)
    reward = sfs.calculate_reward_from_context(result, ctx)
    assert np.isfinite(reward)

def test_compute_pi_from_afs():
    """Test if pi from AFS returns finite float"""
    afs = np.zeros(22)
    afs[1:5] = [10, 5, 3, 1]
    result = pi.compute_pi_from_afs(afs)
    assert isinstance(result, float)
    assert np.isfinite(result)

def test_compute_pi_from_afs_zero():
    """Test if zero AFS yields zero pi"""
    afs = np.zeros(22)
    result = pi.compute_pi_from_afs(afs)
    assert result == 0.0

def test_process_state_pi_shape():
    """Test if  processed pi state has expected stack shape"""
    afs = np.zeros(22)
    afs[1:5] = [10, 5, 3, 1]
    obs, ctx = pi.get_initial_state(target=afs)
    result = pi.process_state(afs, ctx, step_count=2)
    assert result.shape == (pi.STACK_SIZE, 1)

def test_calculate_reward_pi_finite():
    """Test if pi reward computation returns finite value"""
    afs = np.zeros(22)
    afs[1:5] = [10, 5, 3, 1]
    obs, ctx = pi.get_initial_state(target=afs)
    result = pi.process_state(afs, ctx, step_count=2)
    reward = pi.calculate_reward_from_context(result, ctx)
    assert np.isfinite(reward)

def test_get_sfs_from_ms_shape():
    """Test if MS-formatted input yields correct SFS bin shape"""
    ms_data = "//\nsegsites: 3\npositions: 0.1 0.5 0.9\n110\n010\n001\n"
    result = sfs.get_sfs_from_ms(ms_data)
    assert result.shape == (sfs.NUM_BINS_MS,)

def test_compute_pi_from_ms():
    """Test if pi from MS-formatted input returns finite float"""
    ms_data = "//\nsegsites: 3\npositions: 0.1 0.5 0.9\n110\n010\n001\n"
    result = pi.compute_pi_from_ms(ms_data)
    assert isinstance(result, float)
    assert np.isfinite(result)