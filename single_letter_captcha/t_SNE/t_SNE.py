import numpy as np
from matplotlib import pyplot as plt
from tsne import bh_sne

import load_pickle_database

[dataset_read,label_read] = load_pickle_database.load_images_labels("/Users/yiliu/Box Sync/Stanford/Classes/2017-2018 Fall/CS 229 Machine Learning/Project/CAPTCHA Database/Simple single letter dataset/50000_single_letter.p")
print(dataset_read.shape)
print(label_read.shape)

x_data = np.asarray(dataset_read).astype('float64')
x_data = x_data.reshape((x_data.shape[0], -1))
y_data = label_read

vis_data = bh_sne(x_data,perplexity=10)

vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 26))
plt.colorbar(ticks=range(26))
plt.clim(-0.5, 25.5)
plt.show()
