# import packages
import numpy as np
from matplotlib import pyplot as plt

# total 3 bars (MNIST, SVHN, CIFAR10)
N = 3

# setup color names
color1 = 'firebrick'
color2 = 'steelblue'
color3 = 'darkcyan'
color4 = 'darkorange'

# test accuracy obtained from `test.py`,
# bar1 is the model been binarized, while
# bar2 is the 32-bit floating-point model
bar1 = [98.04, 83.02, 62.94]
bar2 = [98.52, 86.42, 63.62]

# setup figure,
fig = plt.figure(dpi=120, figsize=[6, 5])
plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
plt.rc("ytick", labelsize=14)
# only one plot here
ax = plt.subplot(111)
# setup grid in y axis for better visualization
ax.grid('on', axis='y', linestyle='-.', zorder=0)
ind = np.arange(1, N+1)    # the x locations for the groups
width = 0.3         # the width of the bars
# plot histogram
p1 = ax.bar([i-0.5*width for i in ind], bar1, width, bottom=0, zorder=3, color=color2, alpha=0.8)
p2 = ax.bar([i+0.5*width for i in ind], bar2, width, zorder=3, color=color1, alpha=0.6)
# remove top right axis for better results
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position('zero')

# plt.setp(ax.get_xticklabels(), visible=False)
ax.tick_params(axis='x', which='both', length=0)
# setup x axis tick label
ax.set_xticks(range(1, 4))
ax.set_xticklabels(['MNIST', 'SVHN', 'CIFAR10'])
ax.set_xlim(0.5, 3.8)
ax.set_ylabel('Test Accuracy', fontsize=17)
ax.legend((p1[0], p2[0]), ('Binary', 'Full Precision'), prop={'size': 14})
plt.show()