import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def setup_logger(
    logger_name: str = "telegram_assist",
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "app.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    )

    # File handler: ротация по датам
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
