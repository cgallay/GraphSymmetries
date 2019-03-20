from tensorboardX import SummaryWriter

class Logger():
    def __init__(self, log_dir):
        self.tensorboard = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def write_graph(self, model, sample):
        try:
            self.tensorboard.add_graph(model, sample, False)
        except:
            print("Graph could not be loged in tensordboard.")

    def write(self, metrics, curve='train', increment=True):
        """
        log the loss and the accuracy
        """
        for key, value in metrics.items():
            self.tensorboard.add_scalars(f'metrics/{key}', {curve: value}, self.step)
        if increment:
            self.step += 1

