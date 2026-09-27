"""HTTP service for the Cali cadastral address normalizer.

``cali_address.api.main:app`` is the ASGI application; it loads the model and
the gazetteer once at startup and runs exactly the same engine as
``scripts/normalizar.py`` (``cali_address.service``).
"""

from __future__ import annotations

__all__ = ["app"]


def __getattr__(name):
    # Lazy so `import cali_address.api` does not pull in fastapi/torch.
    if name == "app":
        from .main import app

        return app
    raise AttributeError(name)
