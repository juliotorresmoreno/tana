
class NotImplementedMethodError(Exception):
    def __init__(self, method_name):
        super().__init__(f"The method '{method_name}' is not implemented.")
