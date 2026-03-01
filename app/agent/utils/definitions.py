import json

from pygents import ToolRegistry


def get_tools_definitions() -> str:
    return json.dumps(ToolRegistry.definitions(), indent=2)
