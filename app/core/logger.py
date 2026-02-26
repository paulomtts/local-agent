from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING

from app.core.config import ENVIRONMENT

if TYPE_CHECKING:
    from py_ai_toolkit.core.domain.interfaces import CompletionResponse

LEVEL_COLORS = {
    "DEBUG": "\033[90m",
    "INFO": "\033[92m",
    "WARNING": "\033[93m",
    "ERROR": "\033[91m",
    "CRITICAL": "\033[95m",
}

RESET = "\033[0m"  # Reset color
TASK_TAG = "\033[93m"  # Yellow
HOOK_TAG = "\033[96m"  # Dark cyan
PROMPT_TAG = "\033[38;5;117m"  # Light cyan/sky blue
TOOL_TAG = "\033[38;5;208m"  # Dark orange
SUBTOOL_TAG = "\033[38;5;215m"  # Light orange


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
if ENVIRONMENT == "local":
    logger.setLevel(DEBUG)
logger.addHandler(handler)


def log_hook(hook_name: str, action: str, detail: str = "") -> None:
    msg = f"{HOOK_TAG}[HOOK:{hook_name}:{action}]{RESET}"
    if detail:
        msg += f" {detail}"
    logger.debug(msg)


def log_task(task_name: str, detail: str = "") -> None:
    msg = f"{TASK_TAG}[TASK:{task_name}]{RESET}"
    if detail:
        msg += f" {detail}"
    logger.debug(msg)


def log_tool_use(tool_name: str) -> None:
    logger.debug(f"{TOOL_TAG}[TOOL:{tool_name}]{RESET}")


def log_tool_subtool_use(tool_name: str, subtool_name: str) -> None:
    logger.debug(f"{TOOL_TAG}[TOOL:{tool_name}:{subtool_name}]{RESET}")


def log_token_usage(prompt_name: str, response: "CompletionResponse") -> None:
    """Log prompt token usage from CompletionResponse."""
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    completion = response.completion
    if (
        isinstance(completion, (ChatCompletion, ChatCompletionChunk))
        and completion.usage
    ):
        usage = completion.usage
        logger.debug(
            f"{PROMPT_TAG}[PROMPT:{prompt_name}]{RESET} "
            f"in={usage.prompt_tokens} out={usage.completion_tokens} total={usage.total_tokens}"
        )
