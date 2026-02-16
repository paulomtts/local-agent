from logging import INFO, Formatter, StreamHandler, getLogger

LEVEL_COLORS = {
    "DEBUG": "\033[90m",
    "INFO": "\033[92m",
    "WARNING": "\033[93m",
    "ERROR": "\033[91m",
    "CRITICAL": "\033[95m",
}

RESET = "\033[0m"


class ColoredFormatter(Formatter):
    def format(self, record):
        level_color = LEVEL_COLORS.get(record.levelname, RESET)
        record.levelname = f"{level_color}{record.levelname}{RESET}"
        return super().format(record)


handler = StreamHandler()
handler.setFormatter(
    ColoredFormatter(
        "\033[90m%(asctime)s\033[0m | %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)

logger = getLogger("local_agent")
logger.setLevel(INFO)
logger.addHandler(handler)
