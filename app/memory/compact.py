from pygents import ContextItem, ContextQueue

from app.core.factories import get_toolkit
from app.core.logger import log_task, log_token_usage
from app.memory.dataclasses import Compaction

COMPACT_PROMPT = """Summarize the following conversation items into a single consolidated summary. Preserve all information that is relevant for the assistant to continue helping the user (recent intents, file contents or paths mentioned, decisions, tool outcomes). Omit only redundant or purely decorative detail. Output the summary only, no preamble.

# Working Memory (Items to Compact)
{{ old_items }}
"""


async def compact_memory(memory: ContextQueue, items: list[ContextItem]):
    """Compact older memory items into a summary, keeping recent items intact.

    Args:
        memory: Context queue to append the compaction result to.
        items: List of memory items (MemoryItemType objects)
    """
    KEEP_RECENT = 5

    if len(items) <= KEEP_RECENT:
        return

    old_items = items[:-KEEP_RECENT]
    recent_items = items[-KEEP_RECENT:]
    if len(old_items) == KEEP_RECENT:
        old_items = items
        recent_items = []

    log_task("compact_memory", f"{len(old_items)}→{len(recent_items)} items")

    toolkit = get_toolkit()
    old_items_text = "\n".join(str(item.content) for item in old_items)
    output = await toolkit.chat(
        template=COMPACT_PROMPT,
        old_items=old_items_text,
    )

    log_token_usage("compact", output)

    compacted = Compaction(summary=output.content, items_compacted=len(old_items))
    compaction_item = ContextItem(compacted)
    from app.memory.hooks import LOADING_WORKING_MEMORY

    token = LOADING_WORKING_MEMORY.set(True)
    try:
        await memory.clear()
        await memory.append(compaction_item)
        for item in recent_items:
            await memory.append(item)
    finally:
        LOADING_WORKING_MEMORY.reset(token)
