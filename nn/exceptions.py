class NetworkException(Exception):
    def __init__(self, msg=''):
        super().__init__(msg)


class LayerConnectingException(NetworkException):
    def __init__(self, layer_no, layer):
        super().__init__(f'Exception occurred during connecting layer {layer_no} ({str(layer)})')


class PropagationException(NetworkException):
    def __init__(self, layer_no, layer):
        super().__init__(f'Exception occurred during propagation through layer {layer_no} ({str(layer)})')


class BackpropagationException(NetworkException):
    def __init__(self, layer_no, layer):
        super().__init__(f'Exception occurred during backpropagation through layer {layer_no} ({str(layer)})')


class InvalidShapeException(NetworkException):
    def __init__(self, msg=''):
        super().__init__(msg)


class InvalidLayerPositionException(NetworkException):
    def __init__(self, msg=''):
        super().__init__(msg)


class InvalidParameterException(NetworkException):
    def __init__(self, msg=''):
        super().__init__(msg)


class InvalidLabelsException(NetworkException):
    def __init__(self, msg=''):
        super().__init__(msg)
