"""Owned process groups backed by Windows Job Objects.

All direct Win32 calls live in :class:`Win32JobApi`.  ``NoopJobApi`` is a
portable adapter for tests and non-Windows development; it owns only the
processes it starts and never searches for processes by name.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import math
import os
import signal
import subprocess
import threading
from typing import Dict, List, Optional


JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
CREATE_SUSPENDED = 0x00000004
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_UNICODE_ENVIRONMENT = 0x00000400
CTRL_BREAK_EVENT = 1
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258
INFINITE = 0xFFFFFFFF


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(wintypes.BYTE)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class _PopenProcess:
    def __init__(self, process: subprocess.Popen):
        self._process = process
        self.pid = process.pid
        self.native_handle = getattr(process, "_handle", process.pid)

    def is_running(self):
        return self._process.poll() is None

    def send_graceful(self):
        if not self.is_running():
            return
        try:
            if os.name == "nt":
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(self.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    def wait(self, timeout):
        return self._process.wait(timeout=timeout)

    def force_stop(self):
        if not self.is_running():
            return
        try:
            if os.name == "nt":
                self._process.kill()
            else:
                os.killpg(self.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass

    def close(self):
        # Popen closes its native process handle when the object is finalized.
        pass


class _NoopJobHandle:
    pass


class NoopJobApi:
    """Portable test adapter with no system-wide process lookup."""

    def __init__(self):
        self._processes: Dict[object, List[_PopenProcess]] = {}
        self._kill_on_close: Dict[object, bool] = {}

    def create_job(self):
        handle = _NoopJobHandle()
        self._processes[handle] = []
        self._kill_on_close[handle] = False
        return handle

    def set_kill_on_close(self, job_handle):
        self._kill_on_close[job_handle] = True

    def assign_current_process(self, job_handle):
        # Portable no-op: there is no real, system-wide job membership to
        # establish outside a real Windows Job Object, and -- critically --
        # this adapter's own `close_handle`/`terminate_job` kill-on-close
        # path only ever force-stops processes *it itself started* via
        # `start_assigned` (tracked in `self._processes`). It must never
        # register the calling (test) process as one of those, or a test
        # that closes a kill-on-close job after assigning itself to it
        # would force-stop its own test runner.
        pass

    def start_assigned(self, job_handle, spec):
        kwargs = {
            "cwd": str(spec.cwd) if spec.cwd is not None else None,
            "env": dict(spec.env) if spec.env is not None else None,
        }
        if os.name == "nt":
            kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        process = _PopenProcess(subprocess.Popen(list(spec.argv), **kwargs))
        self._processes.setdefault(job_handle, []).append(process)
        return process

    def terminate_job(self, job_handle, exit_code):
        del exit_code
        for process in self._processes.get(job_handle, ()):
            process.force_stop()

    def close_handle(self, handle):
        if self._kill_on_close.get(handle):
            self.terminate_job(handle, 1)
        self._processes.pop(handle, None)
        self._kill_on_close.pop(handle, None)


class _Win32Process:
    def __init__(self, api: "Win32JobApi", process_handle, pid: int):
        self._api = api
        self._handle = process_handle
        self.native_handle = process_handle
        self.pid = pid
        self._returncode: Optional[int] = None
        self._lock = threading.RLock()

    def _finish(self):
        if self._returncode is None:
            self._returncode = self._api.get_exit_code(self._handle)
        self.close()
        return self._returncode

    def is_running(self):
        with self._lock:
            if self._returncode is not None or self._handle is None:
                return False
            result = self._api.wait_for_process(self._handle, 0)
            if result == WAIT_TIMEOUT:
                return True
            if result != WAIT_OBJECT_0:
                raise OSError(f"WaitForSingleObject returned unexpected status {result}")
            self._finish()
            return False

    def send_graceful(self):
        if self.is_running():
            self._api.generate_ctrl_break(self.pid)

    def wait(self, timeout):
        with self._lock:
            if self._returncode is not None:
                return self._returncode
            milliseconds = INFINITE if timeout is None else max(0, math.ceil(timeout * 1000))
            result = self._api.wait_for_process(self._handle, milliseconds)
            if result == WAIT_TIMEOUT:
                raise subprocess.TimeoutExpired(["owned-process", str(self.pid)], timeout)
            if result != WAIT_OBJECT_0:
                raise OSError(f"WaitForSingleObject returned unexpected status {result}")
            return self._finish()

    def force_stop(self):
        with self._lock:
            if self._handle is not None and self.is_running():
                self._api.terminate_process(self._handle, 1)

    def close(self):
        with self._lock:
            if self._handle is not None:
                self._api.close_handle(self._handle)
                self._handle = None
                self.native_handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Win32JobApi:
    """Small ctypes adapter for Job Object and suspended process APIs."""

    def __init__(self):
        if os.name != "nt":
            raise OSError("Win32JobApi is available only on Windows")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_signatures()

    def _configure_signatures(self):
        kernel32 = self._kernel32
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CreateProcessW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.LPCWSTR,
            ctypes.POINTER(_StartupInfo),
            ctypes.POINTER(_ProcessInformation),
        ]
        kernel32.CreateProcessW.restype = wintypes.BOOL
        kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
        kernel32.ResumeThread.restype = wintypes.DWORD
        kernel32.GenerateConsoleCtrlEvent.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.GenerateConsoleCtrlEvent.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

    @staticmethod
    def _raise_last_error(api_name):
        raise ctypes.WinError(ctypes.get_last_error(), api_name)

    def create_job(self):
        handle = self._kernel32.CreateJobObjectW(None, None)
        if not handle:
            self._raise_last_error("CreateJobObjectW")
        return handle

    def set_kill_on_close(self, job_handle):
        limits = _JobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self._kernel32.SetInformationJobObject(
            job_handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            self._raise_last_error("SetInformationJobObject")

    @staticmethod
    def _environment_block(environment):
        if environment is None:
            return None
        entries = [f"{key}={value}" for key, value in sorted(environment.items(), key=lambda item: item[0].upper())]
        return ctypes.create_unicode_buffer("\0".join(entries) + "\0\0")

    def start_assigned(self, job_handle, spec):
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(spec.argv)))
        environment = self._environment_block(spec.env)
        startup_info = _StartupInfo()
        startup_info.cb = ctypes.sizeof(startup_info)
        process_info = _ProcessInformation()
        creation_flags = CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP
        if environment is not None:
            creation_flags |= CREATE_UNICODE_ENVIRONMENT
        if not self._kernel32.CreateProcessW(
            None,
            command_line,
            None,
            None,
            False,
            creation_flags,
            ctypes.cast(environment, ctypes.c_void_p) if environment is not None else None,
            str(spec.cwd) if spec.cwd is not None else None,
            ctypes.byref(startup_info),
            ctypes.byref(process_info),
        ):
            self._raise_last_error("CreateProcessW")

        try:
            if not self._kernel32.AssignProcessToJobObject(job_handle, process_info.hProcess):
                self._raise_last_error("AssignProcessToJobObject")
            previous_suspend_count = self._kernel32.ResumeThread(process_info.hThread)
            if previous_suspend_count == 0xFFFFFFFF:
                self._raise_last_error("ResumeThread")
        except BaseException:
            self._cleanup_failed_start(process_info, close_thread=True)
            raise

        try:
            self.close_handle(process_info.hThread)
        except BaseException:
            # The process has already resumed.  Do not let failure to release
            # its thread handle orphan a running child with no returned owner.
            self._cleanup_failed_start(process_info, close_thread=False)
            raise

        return _Win32Process(self, process_info.hProcess, int(process_info.dwProcessId))

    def _cleanup_failed_start(self, process_info, close_thread):
        try:
            self.terminate_process(process_info.hProcess, 1)
        except OSError:
            pass
        if close_thread:
            try:
                self.close_handle(process_info.hThread)
            except OSError:
                pass
        try:
            self.close_handle(process_info.hProcess)
        except OSError:
            pass

    def assign_current_process(self, job_handle):
        """Assign the *calling* process (this Python process) to
        `job_handle` -- the "attach an already-running process" primitive
        `WindowsJob`'s existing surface never needed until now (every other
        caller starts a brand-new process already assigned via
        `start_assigned`/`CreateProcessW`'s suspended-then-assign dance).
        `GetCurrentProcess()` returns a pseudo-handle that never needs
        closing (it is not a real handle slot)."""
        current_process = self._kernel32.GetCurrentProcess()
        if not self._kernel32.AssignProcessToJobObject(job_handle, current_process):
            self._raise_last_error("AssignProcessToJobObject")

    def terminate_job(self, job_handle, exit_code):
        if not self._kernel32.TerminateJobObject(job_handle, exit_code):
            self._raise_last_error("TerminateJobObject")

    def wait_for_process(self, process_handle, milliseconds):
        return int(self._kernel32.WaitForSingleObject(process_handle, milliseconds))

    def get_exit_code(self, process_handle):
        exit_code = wintypes.DWORD()
        if not self._kernel32.GetExitCodeProcess(process_handle, ctypes.byref(exit_code)):
            self._raise_last_error("GetExitCodeProcess")
        return int(exit_code.value)

    def generate_ctrl_break(self, process_group_id):
        return bool(self._kernel32.GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, process_group_id))

    def terminate_process(self, process_handle, exit_code):
        if not self._kernel32.TerminateProcess(process_handle, exit_code):
            self._raise_last_error("TerminateProcess")

    def close_handle(self, handle):
        if handle and not self._kernel32.CloseHandle(handle):
            self._raise_last_error("CloseHandle")


class WindowsJob:
    """A private job whose handle bounds only processes started through it."""

    def __init__(self, kill_on_close=True, adapter=None):
        self._api = adapter if adapter is not None else (Win32JobApi() if os.name == "nt" else NoopJobApi())
        self._lock = threading.RLock()
        self._handle = self._api.create_job()
        self._assigned_pids = []
        if kill_on_close:
            try:
                self._api.set_kill_on_close(self._handle)
            except BaseException:
                self._api.close_handle(self._handle)
                self._handle = None
                raise

    @property
    def assigned_pids(self):
        with self._lock:
            return tuple(self._assigned_pids)

    def start_assigned(self, spec):
        with self._lock:
            if self._handle is None:
                raise RuntimeError("cannot start a process in a closed job")
            process = self._api.start_assigned(self._handle, spec)
            self._assigned_pids.append(process.pid)
            return process

    def assign_current_process(self):
        """Assign the calling process itself (not a child) to this job.

        With `kill_on_close=True` (this class's default), every descendant
        this process spawns anywhere in its tree -- including a plain
        `subprocess.Popen`/`subprocess.run` call with no job awareness at
        all, such as the `ffmpeg`/`ffprobe` calls `visual_connector.py`
        makes -- inherits job membership automatically per normal Windows
        job semantics (a child process is placed in its creator's job
        unless the creator explicitly opts out), so closing/terminating
        this job becomes a real backstop that reaches process trees this
        module was never told about, not only processes started through
        `start_assigned`.
        """
        with self._lock:
            if self._handle is None:
                raise RuntimeError("cannot assign to a closed job")
            self._api.assign_current_process(self._handle)
            self._assigned_pids.append(os.getpid())

    def terminate(self, exit_code=1):
        with self._lock:
            if self._handle is None:
                return False
            self._api.terminate_job(self._handle, exit_code)
            return True

    def close(self):
        with self._lock:
            if self._handle is not None:
                self._api.close_handle(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
