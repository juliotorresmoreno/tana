from abc import abstractmethod
from pipe.Pipeline import Node
from common.errors import NotImplementedMethodError

class Library(Node):
    task = 'search'

    def __init__(self):
        super().__init__(self.task)

    @abstractmethod
    def create_index(self, index_name: str):
        raise NotImplementedMethodError("create_index")

    @abstractmethod
    def add(self, index_name: str, data_array: list):
        raise NotImplementedMethodError("create_index")
    
    @abstractmethod
    def invoke(self, index_name: str, text: str):
        raise NotImplementedMethodError("create_index")