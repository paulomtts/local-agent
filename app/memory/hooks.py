import sys

from pygents import ContextItem, ContextQueueHook, hook

from app.core.logger import HOOK_TAG, RESET, logger
from app.memory.working import build_tree, write_working_memory


@hook(ContextQueueHook.AFTER_APPEND)
async def after_append(
    _queue, _appended_items: list[ContextItem], current: list[ContextItem]
):
    sys.stdout.write("\n")
    sys.stdout.flush()
    tree = build_tree(current, appended_items=_appended_items)
    if tree is None:
        logger.debug(f"{HOOK_TAG}[HOOK:after_append:skip]{RESET}")
        write_working_memory(current, added=1)
        return

    try:
        await tree.run()
    except Exception as e:
        logger.error(f"{HOOK_TAG}[HOOK:after_append:error]{RESET} {e}")

    write_working_memory(current, added=1)
