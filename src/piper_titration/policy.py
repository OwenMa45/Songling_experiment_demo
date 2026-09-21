"""Bounded openpi websocket calls, with no infinite retry loop."""
from collections import deque
import numpy as np


class RemotePolicy:
    def __init__(self, uri, timeout=10, execute_steps=4):
        from websockets.sync.client import connect
        from openpi_client import msgpack_numpy
        if timeout <= 0 or execute_steps < 1:
            raise ValueError("timeout and execute_steps must be positive")
        self.codec = msgpack_numpy
        self.timeout = timeout
        self.execute_steps = execute_steps
        self.queue = deque()
        self.ws = connect(uri, compression=None, max_size=64*1024*1024, open_timeout=timeout, close_timeout=2)
        try:
            self.metadata = self.codec.unpackb(self.ws.recv(timeout=timeout))
            if self.metadata.get("action_schema") != "piper-titration-v1" or self.metadata.get("action_dim") != 14:
                raise ValueError("Server does not declare the PiPER action contract")
        except Exception:
            self.ws.close()
            raise

    def action(self, observation):
        if not self.queue:
            self.ws.send(self.codec.Packer().pack(observation))
            reply = self.ws.recv(timeout=self.timeout)
            if isinstance(reply, str):
                raise RuntimeError(reply)
            actions = np.asarray(self.codec.unpackb(reply)["actions"])
            if actions.ndim != 2 or actions.shape[1] != 14 or not len(actions) or not np.isfinite(actions).all():
                raise ValueError("Policy must return finite [horizon,14] absolute targets")
            self.queue.extend(actions[:self.execute_steps])
        return self.queue.popleft()

    def close(self):
        self.queue.clear()
        self.ws.close()
