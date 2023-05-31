import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.loadtxt('embedding_by_try_ano_CSN.txt')
np.multiply(data, 100)
sns.heatmap(data)
plt.show()