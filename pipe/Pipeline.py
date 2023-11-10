from abc import abstractmethod
import time


class Response:
    context: str | None
    result: str | None
    execution_time: float
    required_task: str

    def __init__(
        self,
        context: str = '',
        result: str = '',
        execution_time: float = 0,
        required_task: str = ''
    ) -> None:
        self.context = context
        self.result = result
        self.execution_time = execution_time
        self.required_task = required_task


class Node:
    context: str

    def __init__(self, context: str):
        self.context = context
        self.next = None

    @abstractmethod
    def invoke(self, question: str, context: str | None, task: str | None) -> Response:
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

    def invoke(self, question: str, context: str | None = None, task: str | None = None) -> Response:
        start_time = time.time()
        current = self.first
        current_flow: Response | None = None
        if not self.is_empty():
            while True:
                if current_flow == None:
                    current_flow = current.invoke(question, context, task)
                else:
                    current_flow = current.invoke(
                        question=question,
                        context=current_flow.context,
                        task=current_flow.required_task
                    )
                print(str(current.context) + ' ' +
                      str(current_flow.execution_time), end=" -> ")
                if isinstance(current_flow.result, str) and current_flow.result != '':
                    end_time = time.time()
                    execution_time = end_time - start_time
                    return Response(
                        context=current_flow.context,
                        result=current_flow.result,
                        required_task=current_flow.required_task,
                        execution_time=execution_time
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
            execution_time=execution_time
        )
