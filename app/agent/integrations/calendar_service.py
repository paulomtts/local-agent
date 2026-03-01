"""Mock calendar service integration."""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

CALENDAR_FILE = Path(__file__).resolve().parents[3] / ".tools" / "calendar.md"


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

    @staticmethod
    def create_event(event: CalendarEvent) -> None:
        """Write calendar event to file.

        Args:
            event: CalendarEvent instance to write
        """
        with open(CALENDAR_FILE, "a") as f:
            f.write(event.model_dump_json() + "\n")

    @staticmethod
    def read_events() -> list[CalendarEvent]:
        """Read all calendar events from file, sorted by start time."""
        if not CALENDAR_FILE.exists():
            return []
        events = []
        for line in CALENDAR_FILE.read_text().splitlines():
            if line.strip():
                events.append(CalendarEvent.model_validate_json(line))
        return sorted(events, key=lambda e: e.start_time)

    @staticmethod
    def format_events() -> str:
        events = CalendarService.read_events()
        if not events:
            return "No calendar events found."

        lines = []
        for event in events:
            start = event.start_time.strftime("%Y-%m-%d %H:%M")
            end = event.end_time.strftime("%H:%M")
            lines.append(f"- [{event.event_id}] {event.title} ({start} → {end})")
            if event.description:
                lines.append(f"  {event.description}")
        return "\n".join(lines)
