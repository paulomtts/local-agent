import pty
import os
import time
from app.sessions import TerminalSession


def get_terminal_session() -> TerminalSession:
    master_fd, slave_fd = pty.openpty()
    pid = os.fork()

    if pid == 0:
        os.close(master_fd)
        os.setsid()
        os.dup2(slave_fd, 0)
        os.dup2(slave_fd, 1)
        os.dup2(slave_fd, 2)
        os.close(slave_fd)
        os.chdir(os.getcwd())
        os.execvp("bash", ["bash", "--norc", "--noprofile"])
    else:
        os.close(slave_fd)
        session = TerminalSession(cwd=os.getcwd(), master_fd=master_fd, pid=pid)
        time.sleep(0.1)
        session.read(timeout=0.5)
        session.write(b"stty -echo; PS1=''; PS2=''\n")
        time.sleep(0.1)
        session.read(timeout=0.5)
        return session
