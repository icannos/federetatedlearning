import pandas as pd
import matplotlib.pyplot as plt

PATH = "benchmarks/unbalanced--mean_benchmark_mnist-3000-512-1.csv"

d = pd.read_csv(PATH)

columns = ['collab_loss', 'collab_auc', 'agent0_loss',
       'agen0_accuracy', 'agen0_auc', 'agent1_loss', 'agen1_accuracy',
       'agen1_auc']

data = d.as_matrix(columns=['collab_accuracy', "agen0_accuracy", "agen1_accuracy"])


""""""
X = [i for i in range(0, data.shape[0])]

plt.plot(X, data[:, 0], label="Collaborative acc")
plt.plot(X, data[:, 1], label="Agent 1 -- acc")
plt.plot(X, data[:, 2], label="Agent 2 -- acc")
#plt.plot(X, data[:, 3], label="Agent 3 -- acc")

plt.legend(loc='lower right')

plt.savefig(f"img/unbalanced.png", format="png")
