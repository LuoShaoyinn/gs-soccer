from abc import ABC, abstractmethod


class Algorithm(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        raise NotImplementedError
