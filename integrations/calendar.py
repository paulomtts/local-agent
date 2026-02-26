"""Mock calendar service integration."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

CALENDAR_FILE = Path(__file__).resolve().parents[1] / ".memory" / "calendar.md"


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

    def __init__(self):
        """Initialize calendar service.

        Args:
            calendar_file: Path to the calendar file in root directory
        """
        self.calendar_file = CALENDAR_FILE

    def create_event(self, event: CalendarEvent) -> None:
        """Write calendar event to file.

        Args:
            event: CalendarEvent instance to write
        """
        with open(self.calendar_file, "a") as f:
            f.write(event.model_dump_json() + "\n")

    def read_events(self) -> list[CalendarEvent]:
        """Read all calendar events from file, sorted by start time."""
        if not self.calendar_file.exists():
            return []
        events = []
        for line in self.calendar_file.read_text().splitlines():
            if line.strip():
                events.append(CalendarEvent.model_validate_json(line))
        return sorted(events, key=lambda e: e.start_time)
