from abc import ABC, abstractmethod


class Module(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args, **kwargs):
        raise NotImplementedError
