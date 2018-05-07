import numpy as np
import matplotlib.pyplot as plt
class Plotter(object):
    def __init__(self, config, train_name, val_name):
        self.train_list = []
        self.train_list_epoch = []
        self.val_list = []
        self.val_list_epoch = []
        self.total_iterations = []
        self.epoch_iterations = []
        self.train_name = train_name
        self.val_name = val_name

    def update_train_lists(self, iteration_count, train_item):
        self.train_list.append(train_item)
        self.total_iterations.append(iteration_count)

    def update_val_lists(self, val_item):
        self.val_list.append(val_item)

    def upate_train_lists_epoch(self, epoch_count, train_item):
        self.epoch_iterations.append(epoch_count)
        self.train_list_epoch.append(train_item)

    def update_val_lists_epoch(self, val_item):
        self.val_list_epoch.append(val_item)

    def add_to_plot(self, ax, lines, labels, X, y, label):
        line, = ax.plot(X, y, label=label)
        lines.append(line)
        labels.append(line.get_label())
        return lines, labels

    def plot(self, plot_dir):
        lines = []
        labels = []
        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(111)
        #ax11 = ax1.twinx()
        numpy_total_iteration_list = np.asarray(self.total_iterations)
        numpy_epoch_iteration_list = np.asarray(self.epoch_iterations)

        lines, labels = self.add_to_plot(ax1, lines, labels, numpy_total_iteration_list,
                                         np.asarray(self.train_list), self.train_name)
        if self.train_list_epoch:
            lines, labels = self.add_to_plot(ax1, lines, labels, numpy_epoch_iteration_list,
                                             np.asarray(self.train_list_epoch), self.train_name+'_epoch')
        if self.val_list:
            lines, labels = self.add_to_plot(ax1, lines, labels, numpy_total_iteration_list,
                                             np.asarray(self.val_list), self.val_name)
        if self.val_list_epoch:
            lines, labels = self.add_to_plot(ax1, lines, labels, numpy_epoch_iteration_list,
                                             np.asarray(self.val_list_epoch), self.val_name+'_epoch')

        ax1.legend(lines, labels)
        print('plot name', plot_dir)
        plt.savefig(plot_dir)

