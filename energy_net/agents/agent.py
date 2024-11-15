from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def get_action(self, observation, deterministic=False):
        pass

    @abstractmethod
    def train(self):
        pass
    @abstractmethod
    def eval(self):
        pass

