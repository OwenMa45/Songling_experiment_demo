from collections import deque
import unittest
import numpy as np
from piper_titration.policy import RemotePolicy


class FakeCodec:
    @staticmethod
    def unpackb(value):
        return value

    class Packer:
        def pack(self, value):
            return value


class Socket:
    def __init__(self, reply):
        self.reply = reply
        self.sent = []
        self.timeout = None

    def send(self, data):
        self.sent.append(data)

    def recv(self, timeout):
        self.timeout = timeout
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class PolicyTests(unittest.TestCase):
    def make_policy(self, reply):
        policy = RemotePolicy.__new__(RemotePolicy)
        policy.ws = Socket(reply)
        policy.codec = FakeCodec
        policy.timeout = 2
        policy.execute_steps = 4
        policy.queue = deque()
        return policy

    def test_old_chunk_discarded_after_four_actions(self):
        p = self.make_policy({"actions": np.arange(8*14).reshape(8, 14)})
        for _ in range(5):
            p.action({})
        self.assertEqual(len(p.ws.sent), 2)
        self.assertEqual(p.ws.timeout, 2)

    def test_timeout_propagates_without_retry(self):
        p = self.make_policy(TimeoutError())
        with self.assertRaises(TimeoutError):
            p.action({})
        self.assertEqual(len(p.ws.sent), 1)

    def test_invalid_action_chunk_is_rejected(self):
        for value in (np.zeros((1, 7)), np.full((1, 14), np.nan), np.zeros((0, 14))):
            with self.assertRaises(ValueError):
                self.make_policy({"actions": value}).action({})


if __name__ == "__main__":
    unittest.main()
