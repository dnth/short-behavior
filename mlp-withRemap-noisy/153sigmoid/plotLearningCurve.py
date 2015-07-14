import numpy as np
import matplotlib.pyplot as plt

def plotLearningCurve():
    fig=plt.figure(0, figsize=(10,8) )
#     fig.clf()
#     plt.ioff()
#     plt.subplot(211)
    plt.plot(trn_error[:200], label='Training Set Error', linestyle="--", linewidth=2)
    plt.plot(tst_error[:200], label='Validation Set Error', linewidth=2)
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylim([-0.01,0.17])
    plt.ylabel('MSE')
    plt.legend()
       
#     plt.subplot(212)
#     plt.plot(trn_class_accu, label='Training Set Accuracy', linestyle="--", linewidth=2)
#     plt.plot(tst_class_accu, label='Validation Set Accuracy', linewidth=2)
#     plt.ylim([0,103])
#     plt.ylabel('Percent')
#     plt.xlabel('Epoch')
#     plt.title('Classification Accuracy')
#     plt.legend(loc=4)
       
#     plt.draw()
    plt.tight_layout(pad=2.1)
#     plt.savefig(figPath)
    
trn_error = np.loadtxt("trn_error")
tst_error = np.loadtxt("tst_error")

plotLearningCurve()
plt.show()