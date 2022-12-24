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


class TestCallback:

    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self):
        print('on_epoch_begin')

    def on_epoch_end(self):
        print('on_epoch_end')

    def on_validation_begin(self):
        print('on_validation_begin')

    def on_validation_end(self):
        print('on_validation_end')

    def on_train_begin(self):
        print('on_train_begin')

    def on_train_end(self):
        print('on_train_end')
