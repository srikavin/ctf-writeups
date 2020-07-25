import json
import matplotlib.pyplot as plt
import numpy as np

metrics = json.load(open('metrics.json', 'r'))

batch = np.arange(1, len(metrics['train'])+1)
train = np.array(metrics['train']) * 100
loss = np.array(metrics['loss'])

plt.figure(figsize=[15, 5])
plt.subplot(1, 2, 1)
plt.plot(batch, loss)
plt.title("Training Accuracy")
plt.xlabel("Batch Index")
plt.ylabel("Training Loss")

plt.subplot(1, 2, 2)
plt.plot(batch, train)
plt.title("Training Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Batch Index")

plt.savefig("train.svg", bbox_inches='tight')
plt.show()
