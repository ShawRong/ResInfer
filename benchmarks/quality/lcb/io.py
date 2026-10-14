'Audited stdio compatibility for the pinned LCB executor.'
import io
import os
import sys
import tempfile
from functools import partial
from unittest.mock import patch


def call_method(method, inputs, source_fd=None):
    if isinstance(inputs, list):
        inputs = '\n'.join(inputs)
    # Run inside the executor's isolated process. Bind a real fd 0 so Python
    # handles open(0, 'rb'), stdin.buffer and os.read consistently. On Linux an
    # anonymous memfd also works after upstream disables filesystem deletion.
    source = (os.fdopen(os.dup(source_fd), 'w+b') if source_fd is not None else
              os.fdopen(os.memfd_create('lcb-stdin'), 'w+b')
              if hasattr(os, 'memfd_create') else tempfile.TemporaryFile('w+b'))
    with source:
        source.seek(0)
        source.truncate()
        source.write(inputs.encode('utf-8'))
        source.seek(0)
        saved_stdin = os.dup(0)
        try:
            os.dup2(source.fileno(), 0)
            with os.fdopen(os.dup(0), 'r', encoding='utf-8') as stdin:
                with patch('sys.stdin', stdin):
                    try:
                        return method()
                    except SystemExit:
                        return None
        finally:
            os.dup2(saved_stdin, 0)
            os.close(saved_stdin)


class _CaptureBuffer(io.BytesIO):
    def close(self):
        if not self.closed:
            self.final_value = self.getvalue()
        super().close()


class Capturing(list):
    def __enter__(self):
        self.previous = sys.stdout
        self.buffer = _CaptureBuffer()
        self.stream = io.TextIOWrapper(self.buffer, encoding='utf-8', write_through=True)
        sys.stdout = self.stream
        return self

    def __exit__(self, *args):
        try:
            if not self.stream.closed:
                self.stream.flush()
            value = self.buffer.final_value if self.buffer.closed else self.buffer.getvalue()
            self.append(value.decode('utf-8', errors='replace'))
        finally:
            sys.stdout = self.previous
            self.stream.close()


def install(executor):
    executor.call_method = call_method
    if not hasattr(os, 'memfd_create'):
        # Prepare the unlinked file before run_test's reliability_guard disables
        # unlink/getcwd. Linux uses memfd and needs no persistent backing file.
        executor._resinfer_stdin_file = tempfile.TemporaryFile('w+b')
        executor.call_method = partial(call_method, source_fd=executor._resinfer_stdin_file.fileno())
    executor.Capturing = Capturing
