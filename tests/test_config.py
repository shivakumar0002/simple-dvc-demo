import pytest


class NotInCustomError(Exception):
    def __init__(self, input_, message="Value not in range"):
        self.input_ = input_
        self.message = message
        super().__init__(self.message)

def test_generic():
    a = 2
    b = 2
    assert a == b