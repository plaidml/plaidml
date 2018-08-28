#! /usr/bin/env python
"""
Copyright (c) 2017 Andrew Azarov
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import errno
import logging
import os
import subprocess
import sys
import time

LIN = None
WIN = None
BSD = None
try:
    import fcntl
    LIN = 1
except ImportError:
    # Not *NIX
    pass
try:
    import msvcrt
    WIN = 1
except ImportError:
    # Not Windows
    pass

BSD = hasattr(os, 'O_EXLOCK')


def pid_exists(pid):
    """Check whether pid exists in the current process table."""
    # http://stackoverflow.com/a/23409343/2010538
    # http://stackoverflow.com/a/28065945/2010538
    if os.name != 'nt':
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError as e:
            return e.errno == errno.EPERM
        else:
            return True
    else:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        HANDLE = ctypes.c_void_p
        DWORD = ctypes.c_ulong
        LPDWORD = ctypes.POINTER(DWORD)

        class ExitCodeProcess(ctypes.Structure):
            _fields_ = [('hProcess', HANDLE), ('lpExitCode', LPDWORD)]

        PROCESS_QUERY_INFORMATION = 0x1000
        process = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, 0, pid)
        if not process:
            return False

        ec = ExitCodeProcess()
        out = kernel32.GetExitCodeProcess(process, ctypes.byref(ec))
        if not out:
            err = kernel32.GetLastError()
            if err == 5:  # Look after this change, maybe previous line was just a skip
                # Access is denied.
                logger.warning("Access is denied to get pid info.")
            kernel32.CloseHandle(process)
            return False
        elif bool(ec.lpExitCode):
            # print ec.lpExitCode.contents
            # There is an exit code, it quit
            kernel32.CloseHandle(process)
            return False
        # No exit code, it's running.
        kernel32.CloseHandle(process)
        return True


class Singlet:
    """
    Based on tendo.singleton
    This module provides a Singlet() class which atomically creates a lock file
    containing PID of the running process to prevent parallel execution of the
    same program. In case there is any other instance running already a
    `SingletException` will be thrown. The class will throw `IOError` and
    `OSError` in case there are hardware or OS level corruption.

    >>> from singletony import Singlet
    ... me = Singlet(filename="test.lock", path="/tmp")

    This is helpful for both daemons and simple crontab scripts. Works on *NIX
    and Windows OS's.

    By default this creates a lock file with a filename based on the
    full path to the script file.
    """

    def __init__(self, path):
        self.lockfile = path
        self.pid = str(os.getpid())
        self.fd = None

    def is_running(self):
        logger.info("Singlet lockfile: " + self.lockfile)
        try:
            # If advance UNIX system with atomic locks on open
            if BSD:
                self.fd = os.open(self.lockfile,
                                  os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_EXLOCK | os.O_NONBLOCK)
            else:
                self.fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                if WIN:
                    msvcrt.locking(self.fd, msvcrt.LK_NBRLCK, 65)
                if LIN:
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError) as e:
            if e.errno in (errno.EACCES, errno.EPERM, errno.EWOULDBLOCK):
                self.fd = None
                return True
            elif e.errno == errno.EEXIST:
                # Workaround for Linux/Windows mostly which lacks atomic
                # locking on open
                self.fd = os.open(self.lockfile, os.O_RDWR)
                try:
                    if WIN:
                        msvcrt.locking(self.fd, msvcrt.LK_NBRLCK, 65)
                    if LIN:
                        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except (IOError, OSError) as e:
                    # Some entity has been faster than us if we WOULDBLOCK
                    if e.errno in (errno.EACCES, errno.EPERM, errno.EWOULDBLOCK):
                        self.fd = None
                        return True
                    else:
                        logger.exception("Something went wrong")
                        self.fd = None
                        raise
            else:
                logger.exception("Something went wrong")
                # Anything else is horribly wrong, we need to raise to the
                # upper level so the following code in this try clause
                # won't execute.
                self.fd = None
                raise
        # By this moment the file should be locked or we should exit so
        # there should realistically be no error here, or it can raise
        if self.oldpid_is_running():
            try:
                os.close(self.fd)
            except OSError as e:
                # We shouldn't raise here anything. Because we don't
                # actually care
                logger.exception("Interesting state")
            self.fd = None
            return True
        # Barring any OS/Hardware issue this musn't throw anything. But
        # even if it throws we should raise it because it means we
        # shouldn't run and some serious problem is already happening. So
        # no, I'm not going to escape this one in try/except.
        if hasattr(os, "ftruncate"):
            os.ftruncate(self.fd, 0)  # Erase
        os.lseek(self.fd, 0, 0)  # Rewind
        # Write PID with WIN fix
        os.write(self.fd, self.pid.rjust(64, "#").encode())
        if hasattr(os, "fsync"):
            os.fsync(self.fd)

    def oldpid_is_running(self):
        # Some OS actually recycle pids within same range so we
        # have to check whether oldPid is not the new one so this
        # won't fire up (*BSD).
        # If not and it is still running we'd rather actually exit
        # right here.
        oldPid = os.read(self.fd, 64).strip().lstrip("#".encode())
        return (oldPid and int(oldPid) > 0 and int(oldPid) != int(self.pid) and
                pid_exists(int(oldPid)))

    def __enter__(self):
        while self.is_running():
            # print('Waiting for another instance to complete...')
            time.sleep(5)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # If we are not initialized don't run the clause
        if self.fd:
            try:
                os.close(self.fd)
                os.unlink(self.lockfile)
            except Exception as ex:
                pass


def main():
    lock_path = os.path.normpath(os.path.expanduser('~/.conda_lock'))
    with Singlet(lock_path):
        sys.exit(subprocess.call(['conda'] + sys.argv[1:]))


logger = logging.getLogger("singletony")
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    main()
