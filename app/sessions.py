import os
import select
from dataclasses import dataclass
import time


@dataclass
class TerminalSession:
    cwd: str
    pid: int
    master_fd: int
    _closed: bool = False
    _cmd_counter: int = 0

    def write(self, data: bytes) -> None:
        if self._closed:
            raise ValueError("Session is closed")
        os.write(self.master_fd, data)

    def read(self, timeout: float = 0.1) -> bytes:
        if self._closed:
            return b""
        if select.select([self.master_fd], [], [], timeout)[0]:
            return os.read(self.master_fd, 4096)
        return b""

    def execute(self, command: str, timeout: float = 2.0) -> str:
        self._cmd_counter += 1
        marker = f"__CMD_{os.getpid()}_{self._cmd_counter}__"
        self.write(f"echo {marker}; {command}; echo {marker}\n".encode())

        output = b""
        end_time = time.time() + timeout

        while time.time() < end_time:
            chunk = self.read(timeout=0.1)
            if chunk:
                output += chunk
                decoded = output.decode(errors="replace")
                if decoded.count(marker) >= 2:
                    break

        decoded = output.decode(errors="replace")
        parts = decoded.split(marker)
        if len(parts) >= 3:
            return parts[1].replace("\r", "").strip()
        return ""

    def close(self) -> None:
        if self._closed:
            return
        try:
            os.close(self.master_fd)
        except OSError:
            pass
        try:
            os.waitpid(self.pid, 0)
        except (OSError, ChildProcessError):
            pass
        self._closed = True
