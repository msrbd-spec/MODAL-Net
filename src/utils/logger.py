"""Logging: per-experiment log file + console output."""

import os
import logging
from datetime import datetime


def get_logger(tag: str, mode: str) -> logging.Logger:
    """Create a logger that writes to logs/<tag>/<mode>_<timestamp>.log and console.

    Args:
        tag: Experiment tag (also used as subdirectory name).
        mode: CLI mode name.

    Returns:
        Configured logging.Logger instance.
    """
    log_dir = os.path.join('logs', tag) if tag else 'logs'
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{mode}_{ts}.log')

    logger = logging.getLogger(f'modal.{tag}.{mode}')
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
