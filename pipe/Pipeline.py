import time
from abc import abstractmethod
from decouple import config

env = config('ENV')
if env == None or env == '':
    env = 'development'


class Response:
    context: str | None
    result: str | None
    execution_time: float
    required_task: str
    release: str

    def __init__(
        self,
        context: str = '',
        result: str = '',
        execution_time: float = 0,
        required_task: str = '',
        release: str = ''
    ) -> None:
        self.context = context
        self.result = result
        self.execution_time = execution_time
        self.required_task = required_task
        self.release = release


class Arguments:
    question: str
    context: str | None
    required_task: str | None
    release: str | None

    def __init__(
        self,
        question: str,
        context: str | None = None,
        required_task: str | None = None,
        release: str | None = None
    ) -> None:
        self.question = question
        self.context = context
        self.release = release
        self.required_task = required_task


class Node:
    context: str

    def __init__(self, context: str):
        self.context = context
        self.next = None

    @abstractmethod
    def invoke(self, args: Arguments) -> Response:
        pass


class Pipeline(Node):
    task = 'pipeline'
    first: Node
    last: Node

    def __init__(self):
        super().__init__(self.task)
        self.first = None
        self.last = None

    def is_empty(self):
        return self.first is None

    def insert(self, new_node):
        if self.is_empty():
            self.first = new_node
            self.last = new_node
        else:
            self.last.next = new_node
            self.last = new_node
            self.last.next = self.first

    def delete(self, node):
        if not self.is_empty():
            current = self.first
            previous = None
            while current:
                if current == node:
                    if previous:
                        previous.next = current.next
                        if current == self.first:
                            self.first = current.next
                        if current == self.last:
                            self.last = previous
                    else:
                        self.first = current.next
                        self.last.next = self.first
                    return
                previous = current
                current = current.next
                if current == self.first:
                    break

    def invoke(self, args: Arguments) -> Response:
        start_time = time.time()
        current = self.first
        cf: Response | None = None
        if not self.is_empty():
            while True:
                ok = cf != None
                cf = current.invoke(Arguments(
                    question=args.question,
                    context=cf.context if ok else args.context,
                    required_task=cf.required_task if ok else args.required_task,
                    release=cf.release if ok else ''
                ))

                if isinstance(cf.result, str) and cf.result != '':
                    end_time = time.time()
                    execution_time = end_time - start_time
                    return Response(
                        context=cf.context,
                        result=cf.result,
                        required_task=cf.required_task,
                        execution_time=execution_time,
                        release=cf.release
                    )

                current = current.next
                if current == self.first:
                    break

        end_time = time.time()
        execution_time = end_time - start_time
        return Response(
            context=None,
            result=None,
            required_task=None,
            release=cf.release if cf != None else '',
            execution_time=execution_time
        )
