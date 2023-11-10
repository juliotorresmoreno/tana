
from pipe.Pipeline import Node, Response, Arguments


class Reinformer(Node):
    task = "reinformer"
    finish = False

    def __init__(self, finish: bool):
        super().__init__(self.task)
        self.finish = finish

    def invoke(self, args: Arguments) -> Response:
        if (self.finish):
            return Response(result='I so sorry. I cannot answer this.')
        return Response(execution_time=0, release=args.release)
