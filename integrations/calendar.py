"""Mock calendar service integration."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class CalendarEvent(BaseModel):
    """Response model for calendar event creation."""

    event_id: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f")
    )
    title: str
    start_time: datetime
    end_time: datetime
    description: str = ""


class CalendarService:
    """Mock calendar service for creating events."""

    def __init__(self, events_file: str = "calendar_events.txt"):
        """Initialize calendar service.

        Args:
            events_file: Path to the events file in root directory
        """
        self.events_file = Path(events_file)

    def create_event(self, event: CalendarEvent) -> None:
        """Write calendar event to file.

        Args:
            event: CalendarEvent instance to write
        """
        with open(self.events_file, "a") as f:
            f.write(event.model_dump_json() + "\n")
