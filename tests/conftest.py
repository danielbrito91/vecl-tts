import pytest
from hydra.core.global_hydra import GlobalHydra


@pytest.fixture(autouse=True)
def cleanup_hydra():
    """Fixture to automatically clear Hydra's global state after each test."""
    yield
    GlobalHydra.instance().clear()
