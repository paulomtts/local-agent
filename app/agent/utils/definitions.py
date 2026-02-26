from pygents import ToolRegistry

# Subtools exposed to think so it can route without triggering internal LLM routing.
# key: tool name, value: {subtool_name: description}
SUBTOOLS: dict[str, dict[str, str]] = {
    "calendar": {
        "read": "List the user's upcoming calendar events.",
        "create": "Create a new calendar event.",
    }
}


def get_tools_definitions() -> str:
    definitions = ""
    for tool in ToolRegistry.all():
        definitions += f"{tool.metadata.name}: {tool.metadata.description}\n"
        if tool.metadata.name in SUBTOOLS:
            for index, (subtool_name, subtool_desc) in enumerate(
                SUBTOOLS[tool.metadata.name].items()
            ):
                definitions += f"   {index + 1}. {subtool_name}: {subtool_desc}\n"
    return definitions
