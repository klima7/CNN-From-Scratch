from .. import activations
from .. import losses
from .. import initializers
from .. import metrics
from ..exceptions import InvalidParameterException


def get_activation(activation):
    if isinstance(activation, str):
        return _get_activation_from_name(activation)
    elif isinstance(activation, activations.Activation):
        return activation
    else:
        raise InvalidParameterException(f'Invalid activation: {activation}')


def get_loss(loss):
    if isinstance(loss, str):
        return _get_loss_from_name(loss)
    elif isinstance(loss, losses.Loss):
        return loss
    else:
        raise InvalidParameterException(f'Invalid loss: {loss}')


def get_initializer(initializer):
    if isinstance(initializer, str):
        return _get_initializer_from_name(initializer)
    elif isinstance(initializer, initializers.Initializer):
        return initializer
    else:
        raise InvalidParameterException(f'Invalid initializer: {initializer}')


def get_metric(metric):
    if isinstance(metric, str):
        return _get_metric_from_name(metric)
    elif isinstance(metric, metrics.Metric):
        return metric
    else:
        raise InvalidParameterException(f'Invalid metric: {metric}')


def _get_activation_from_name(name):
    activations_dict = {
        'no': activations.NoActivation,
        'logistic': activations.LogisticActivation,
        'relu': activations.ReLuActivation,
        'leaky_relu': activations.LeakyReLuActivation,
        'tanh': activations.TanhActivation,
        'sin': activations.SinActivation,
    }

    if name not in activations_dict.keys():
        raise InvalidParameterException(f'Unknown activation name: {name}')

    return activations_dict[name]()


def _get_loss_from_name(name):
    losses_dict = {
        'mse': losses.MseLoss,
        'cce': losses.CceLoss,
        'softmax_cce': losses.SoftmaxCceLoss,
    }

    if name not in losses_dict.keys():
        raise InvalidParameterException(f'Unknown loss name: {name}')

    return losses_dict[name]()


def _get_initializer_from_name(name):
    initializers_dict = {
        'constant': initializers.ConstantInitializer,
        'normal': initializers.RandomNormalInitializer,
        'uniform': initializers.RandomUniformInitializer,
        'glorot': initializers.GlorotUniformInitialization,
    }

    if name not in initializers_dict.keys():
        raise InvalidParameterException(f'Unknown initializer name: {name}')

    return initializers_dict[name]()


def _get_metric_from_name(name):
    metrics_list = [metrics.CategoricalAccuracy]

    for metric in metrics_list:
        if metric.NAME == name:
            return metric()

    raise InvalidParameterException(f'Unknown metric name: {name}')
