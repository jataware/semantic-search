from abc import ABC, abstractmethod

class Search(ABC):
    @abstractmethod
    def __init__(self, raw_corpus:list[str]): ...

    @abstractmethod
    def search(self, query:str, n:int=None) -> list[tuple[str, float]]: ...
