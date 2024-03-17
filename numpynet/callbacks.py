class Callback:

    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_validation_begin(self):
        pass

    def on_validation_end(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass


class ModelCheckpoint(Callback):

    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=False, verbose=False):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.best_metric = None

    def on_train_begin(self):
        self.best_metric = self.__get_metric()
        self.__save()

    def on_epoch_end(self):
        if self.save_best_only:
            metric = self.__get_metric()
            if self.__is_metric_better(metric):
                self.__save()
                self.best_metric = metric
        else:
            self.__save()

    def __save(self):
        self.model.save(self.filepath)
        if self.verbose:
            print(f'Saving model to {self.filepath}')

    def __get_metric(self):
        history = self.model.history
        return history[self.monitor][-1] if self.monitor in history else None

    def __is_metric_better(self, metric):
        if self.best_metric is None:
            return True
        if self.mode == 'min':
            return metric < self.best_metric
        elif self.mode == 'max':
            return metric > self.best_metric


class EarlyStopping(Callback):

    def __init__(self, monitor='val_loss', mode='min', patience=0, restore_best_weights=False, verbose=False):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.no_progress_epochs = 0
        self.best_metric = None
        self.best_weights = None

    def on_train_begin(self):
        self.no_progress_epochs = 0
        self.best_metric = self.__get_metric()
        if self.restore_best_weights:
            self.best_weights = self.model.weights

    def on_train_end(self):
        if self.restore_best_weights:
            self.model.weights = self.best_weights

    def on_epoch_end(self):
        metric = self.__get_metric()
        if self.__is_metric_better(metric):
            self.no_progress_epochs = 0
            self.best_metric = metric
            self.best_weights = self.model.weights
        else:
            self.no_progress_epochs += 1

        if self.no_progress_epochs >= self.patience:
            self.model.stop_training = True
            if self.verbose:
                print('Early stopping')

    def __get_metric(self):
        history = self.model.history
        return history[self.monitor][-1] if self.monitor in history else None

    def __is_metric_better(self, metric):
        if self.best_metric is None:
            return True
        if self.mode == 'min':
            return metric < self.best_metric
        elif self.mode == 'max':
            return metric > self.best_metric
