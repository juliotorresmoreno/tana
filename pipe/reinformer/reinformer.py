
from pipe.Pipeline import Node, Response


class Reinformer(Node):
    task = "reinformer"
    finish = False

    def __init__(self, finish: bool):
        super().__init__(self.task)
        self.finish = finish

    def invoke(self, question: str, context: str, task: str):
        if (self.finish):
            return Response(result='I so sorry. I cannot answer this.')
        return Response(execution_time=0)
