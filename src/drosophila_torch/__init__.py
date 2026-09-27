import logging


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def enable_logging(level=logging.INFO):
    """Convenience function to quickly print logs to stdout."""

    # Remove the NullHandler or existing handlers to prevent duplicates
    logger.handlers = []

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="{asctime}.{msecs:0<3.0f} [{levelname}] {name}: {message}",
        datefmt="%Y-%m-%d %H:%M:%S",
        style="{"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)
