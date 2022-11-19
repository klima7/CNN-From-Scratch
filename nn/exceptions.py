class InvalidShapeError(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class InvalidParameterException(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class InvalidLayerPositionException(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class InvalidLabelsException(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)
