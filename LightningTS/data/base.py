from abc import ABC, abstractmethod

class BaseDataIO(ABC):
    def __init__(self, file_path) -> None:
        self.file_path =  file_path
        
    @abstractmethod
    def read(self):
        raise NotImplementedError('A child class of `BaseDataIO` must implement a `read` method.')
    