import numpy as np
import matplotlib.pyplot as plt

lstmValidationAccuracy = np.loadtxt("lstm/20LSTMCell/tst_ClassAccu")
rnnValidationAccuracy = np.loadtxt("rnn/44sigmoid/tst_ClassAccu")
mlpValidationAccuracy = np.loadtxt("mlp/153sigmoid/tst_ClassAccu")
tdnnValidationAccuracy = np.loadtxt("tdnn/25sigmoid/tst_ClassAccu")

lstmValidationLoss = np.loadtxt("lstm/20LSTMCell/tst_error")
rnnValidationLoss = np.loadtxt("rnn/44sigmoid/tst_error")
mlpValidationLoss = np.loadtxt("mlp/153sigmoid/tst_error")
tdnnValidationLoss = np.loadtxt("tdnn/25sigmoid/tst_error")

plt.plot(lstmValidationAccuracy[:300], linewidth=2, label='LSTM')
plt.plot(rnnValidationAccuracy[:300], linewidth=2, label='RNN')
plt.plot(mlpValidationAccuracy[:300], linewidth=2, label='MLP')
plt.plot(tdnnValidationAccuracy[:300], linewidth=2, label='TDNN')
plt.ylabel("Percent (%)")
plt.xlabel("Epoch")
plt.ylim([0,105])
plt.xlim([0,80])
plt.title("Recognition Rate")
plt.legend(loc=4)
plt.grid(True)
plt.show()

plt.plot(lstmValidationLoss[:300], linewidth=2, label='LSTM')
plt.plot(rnnValidationLoss[:300], linewidth=2, label='RNN')
plt.plot(mlpValidationLoss[:300], linewidth=2, label='MLP')
plt.plot(tdnnValidationLoss[:300], linewidth=2, label='TDNN')
plt.ylabel("Error")
plt.xlabel("Epoch")
plt.title("Learning Curve")
plt.ylim([-0.002,0.15])
plt.xlim([0,80])
plt.legend(loc=1)
plt.grid(True)
plt.show()

