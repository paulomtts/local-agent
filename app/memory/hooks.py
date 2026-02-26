from pygents import ContextItem, ContextQueueHook, hook

from app.core.logger import log_hook, logger
from app.memory.working import build_tree, write_working_memory


@hook(ContextQueueHook.AFTER_APPEND)
async def after_append(
    _queue, _appended_items: list[ContextItem], current: list[ContextItem]
):
    tree = build_tree(current, appended_items=_appended_items)
    if tree is None:
        log_hook("after_append", "skip", "extraction")
        write_working_memory(current, added=1)
        return

    try:
        await tree.run()
    except Exception as e:
        logger.error(f"[HOOK:after_append:error] {e}")

    write_working_memory(current, added=1)
