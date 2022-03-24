import pandas as pd
import matplotlib.pyplot as plt

def visualize(log_path):
    log_df = pd.read_csv(log_path)
    epoch = list(log_df['epoch'].unique())
    epochs = [i for i in range(len(epoch))]
    subplot_id = 320
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    for e in epochs:
        e_df = log_df[log_df['epoch'] == e]
        len_e = len(e_df)
        loss_prom = list(e_df['loss_prom'])
        loss_ss = list(e_df['loss_ss'])
        loss_polya = list(e_df['loss_polya'])
        steps = [(k+1) for k in range(0, len_e)]
        subplot_id += 1
        ax = fig.add_subplot(subplot_id)
        ax.plot(steps, loss_prom, label='loss prom')
        ax.plot(steps, loss_ss, label='loss ss')
        ax.plot(steps, loss_polya, label='loss polya')
        ax.set_title('epoch {}'.format(e+1))
        ax.set_xlabel('steps')
        ax.set_ylabel('loss')
        ax.legend()
        
    plt.subplots_adjust(top=1.5, bottom=1, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
    plt.show()

def visualise_average_loss(log_path):
    log_df = pd.read_csv(log_path)
    epochs = log_df["epoch"].unique()
    average_prom = []
    average_ss = []
    average_polya = []
    for e in epochs:
        df = log_df[log_df["epoch"] == e]
        average_prom.append(df["loss_prom"].mean())
        average_ss.append(df["loss_ss"].mean())
        average_polya.append(df["loss_polya"].mean())
    
    figs, axes = plt.subplots(nrows=2, ncols=2)

    axes[0].plot(epochs, average_prom)
    axes[1].plot(epochs, average_ss)
    axes[2].plot(epochs, average_polya)
    
    plt.subplots_adjust(top=1.5, bottom=1, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
    plt.show()