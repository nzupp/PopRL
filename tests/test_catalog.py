"""Unit tests for stdpopsim model retrieval and helpers"""
import pytest
import msprime
from poprl.catalog import get_model, get_model_info

def test_get_model_returns_demography():
    """Test if get_model returns a Demography object"""
    demography, mutation_rate = get_model("HomSap", "Africa_1T12")
    assert isinstance(demography, msprime.Demography)

def test_get_model_returns_mutation_rate():
    """Test if get_model returns positive mutation rate"""
    demography, mutation_rate = get_model("HomSap", "Africa_1T12")
    assert mutation_rate > 0

def test_get_model_invalid_species():
    """Test if get_model returns None tuple for invalid species"""
    result = get_model("FakeSpecies", "FakeModel")
    assert result == (None, None)

def test_get_model_invalid_model():
    """Test if get_model returns None tuple for invalid model"""
    result = get_model("HomSap", "FakeModel")
    assert result == (None, None)

def test_get_model_info_keys():
    """Test if get_model_info returns expected keys"""
    import stdpopsim
    sp = stdpopsim.get_species("HomSap")
    m = sp.get_demographic_model("Africa_1T12")
    info = get_model_info(m)
    assert "id" in info
    assert "mutation_rate" in info
    assert "populations" in info
    assert "events" in info
