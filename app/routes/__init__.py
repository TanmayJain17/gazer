"""Routes package for HTTP and WebSocket handlers."""

from app.routes.websocket_handlers import register_handlers

__all__ = ["register_handlers"]
