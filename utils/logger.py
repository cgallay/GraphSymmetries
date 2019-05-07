from tensorboardX import SummaryWriter

from matplotlib import pyplot as plt

class Logger():
    def __init__(self, log_dir):
        self.tensorboard = SummaryWriter(log_dir=log_dir)
        self.step = 0

    def write_graph(self, model, sample):
        try:
            self.tensorboard.add_graph(model, sample, False)
        except:
            print("Graph could not be loged in tensordboard.")

    def write(self, metrics, curve='train', index=-1, increment=True):
        """
        log the loss and the accuracy
        """
        if index == -1:
            index = self.step

        for key, value in metrics.items():
            self.tensorboard.add_scalars(f'metrics/{key}', {curve: value}, index)
        if increment:
            self.step += 1
    def write_hparam(self, hparam):
        self.tensorboard.add_text('hyperparameters', repr(hparam))
    
    def plot_eploration(self, learning_rates, losses):
        fig=plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.clear()
        ax.set_ylabel('Losses')
        ax.set_xlabel('learning rate')
        ax.plot(learning_rates, losses)
        self.tensorboard.add_figure('learing_rate', fig)
