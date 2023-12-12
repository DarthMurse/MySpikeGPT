import torch
import matplotlib.pyplot as plt
import sys

def plot_curve(name):
    PATH = "ANN_models/" + name + '/'
    loss_curve = torch.load(PATH+"loss_curve")
    valid_loss_curve = torch.load(PATH+"valid_loss_curve")
    time = range(len(loss_curve))
    
    plt.plot(time, valid_loss_curve, color='r', label='valid loss', linewidth=2)
    plt.plot(time, loss_curve, color='b', label='loss', linewidth=2)
    
    plt.legend()
    plt.savefig(PATH+"curve.pdf")
    plt.savefig(PATH+"curve.png", dpi=300)

if __name__ == "__main__":
    plot_curve(sys.argv[1])
