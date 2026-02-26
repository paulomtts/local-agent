from contextvars import ContextVar

from pygents import ContextItem, ContextQueueHook, hook

from app.core.logger import log_hook
from app.memory.working import build_tree, write_working_memory

LOADING_WORKING_MEMORY: ContextVar[bool] = ContextVar(
    "loading_working_memory", default=False
)


@hook(ContextQueueHook.AFTER_APPEND)
async def after_append(
    _queue, _appended_items: list[ContextItem], current: list[ContextItem]
):
    if LOADING_WORKING_MEMORY.get():
        return

    tree = build_tree(current, appended_items=_appended_items, queue=_queue)
    if tree is None:
        log_hook("after_append", "skip", "extraction")
        write_working_memory(current, added=1)
        return

    try:
        await tree.run()
    except Exception as e:
        log_hook("after_append", "error", str(e))

    write_working_memory(list(_queue.items), added=1)
