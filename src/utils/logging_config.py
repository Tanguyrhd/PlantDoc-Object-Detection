import logging

def setup_logging():
    """Parent logging configuration for the project"""

    root = logging.getLogger()

    # Avoid duplication
    if root.handlers:
        return

    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.setLevel(logging.INFO)
