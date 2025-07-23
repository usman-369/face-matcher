import logging

logger = logging.getLogger(__name__)


class FaceMatcherLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra=None):
        if extra is None:
            extra = {}
        super().__init__(logger, extra)

    def process(self, msg, kwargs):
        user_email = self.extra.get("user_email", "unknown")
        return f"[FaceMatcher] ({user_email}) {msg}", kwargs
