"""Small fake objects used by several tests (no network needed)."""

import requests


class FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, text="", json_data=None, status_code=200):
        """Store the pieces a test wants the response to contain."""
        self.text = text
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        """Raise ``requests.HTTPError`` for error status codes."""
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        """Return the JSON payload (or raise ValueError if there is none)."""
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeSession:
    """Fake ``requests.Session`` that answers from a function or a fixed response."""

    def __init__(self, handler):
        """Create it with ``handler(url, params) -> FakeResponse`` (or raise)."""
        self.handler = handler
        self.calls = []

    def get(self, url, params=None, timeout=None):
        """Record the call and return what the handler says."""
        self.calls.append((url, params))
        return self.handler(url, params)
