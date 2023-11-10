from pipe.Pipeline import Node


class ModelBase(Node):
    task = ""
    def __init__(self, data):
        super().__init__(data)
