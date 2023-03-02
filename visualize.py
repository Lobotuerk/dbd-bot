import matplotlib.pyplot as plt
import numpy as np

acc = np.load('./total_acc.npy' , allow_pickle=True)
val_acc = np.load('./total_val_acc.npy' , allow_pickle=True)
loss = np.load('./total_loss.npy' , allow_pickle=True)
val_loss = np.load('./total_val_loss.npy' , allow_pickle=True)

epochs = range(len(acc))
print(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()
