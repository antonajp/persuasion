"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture
def sample_topic():
    """Provide a sample debate topic."""
    return "Carbon pricing policy for climate action"


@pytest.fixture
def sample_goal():
    """Provide a sample persuasion goal."""
    return "Build consensus on carbon pricing policy"
