"""Public compatibility exports for EmbedFlow's existing FastAPI app.

The implementation remains in :mod:`embedflow.serving.api`; this module keeps
the short ``embedflow.api`` import path lightweight and does not create an app
or import FastAPI until :func:`create_app` is called.
"""

from .serving.api import create_app, dashboard_html

__all__ = ["create_app", "dashboard_html"]
