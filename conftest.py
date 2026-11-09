"""Shared pytest setup: every test works on a temporary database and data dir.

Because this file sits in the project root, pytest also adds the root to the
import path, so tests can simply ``import config`` and ``from src import db``.
"""

import pytest

import config


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Point the database and raw-data folders at a temporary directory.

    Args:
        tmp_path: pytest's temporary directory for this test.
        monkeypatch: pytest helper that undoes changes after the test.

    Returns:
        The path of the temporary database file.
    """
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "INDEED_POSTINGS_DIR", tmp_path / "raw" / "indeed")
    monkeypatch.setattr(config, "STATE", "California")
    return config.DB_PATH
