class InvalidShapeError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class InitializerError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class InvalidParameterException(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)
