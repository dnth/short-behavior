import numpy as np
import theanets
import climate
import matplotlib.pyplot as plt
logging = climate.get_logger('lstm-joint')
climate.enable_default_logging()

def plotLearningCurve():
    fig=plt.figure(0, figsize=(10,8) )
    fig.clf()
    plt.plot(trn_loss, label='Training Set Error', linestyle="--", linewidth=2)
    plt.plot(valid_loss, label='Validation Set Error', linewidth=2)
    plt.title('Cross-Entropy Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig("error")
    
# Original Joint data
BallLiftJoint = np.loadtxt('../../../20fpsFullBehaviorSampling/BallLift/JointData.txt').astype(np.float32)
BallRollJoint = np.loadtxt('../../../20fpsFullBehaviorSampling/BallRoll/JointData.txt').astype(np.float32)
BellRingLJoint = np.loadtxt('../../../20fpsFullBehaviorSampling/BellRingL/JointData.txt').astype(np.float32)
BellRingRJoint = np.loadtxt('../../../20fpsFullBehaviorSampling/BellRingR/JointData.txt').astype(np.float32)
BallRollPlateJoint = np.loadtxt('../../../20fpsFullBehaviorSampling/BallRollPlate/JointData.txt').astype(np.float32)
RopewayJoint = np.loadtxt('../../../20fpsFullBehaviorSampling/Ropeway/JointData.txt').astype(np.float32)

# fig=plt.figure(0)
# plt.subplot(121)
# plt.plot(BallLiftJoint[:200,:])

BallLiftJoint = BallLiftJoint.reshape((50,200,10))
BallRollJoint = BallRollJoint.reshape((50,200,10))
BellRingLJoint = BellRingLJoint.reshape((50,200,10))
BellRingRJoint = BellRingRJoint.reshape((50,200,10))
BallRollPlateJoint = BallRollPlateJoint.reshape((50,200,10))
RopewayJoint = RopewayJoint.reshape((50,200,10))

# plt.subplot(122)
# plt.plot(BallLiftJoint[0,:200,:])
# plt.show()

train_series_BL = BallLiftJoint[:40,:,:]
train_series_BR = BallRollJoint[:40,:,:]
train_series_BRL = BellRingLJoint[:40,:,:]
train_series_BRR = BellRingRJoint[:40,:,:]
train_series_BRP = BallRollPlateJoint[:40,:,:]
train_series_RW = RopewayJoint[:40,:,:]

valid_series_BL = BallLiftJoint[40:,:,:]
valid_series_BR = BallRollJoint[40:,:,:]
valid_series_BRL = BellRingLJoint[40:,:,:]
valid_series_BRR = BellRingRJoint[40:,:,:]
valid_series_BRP = BallRollPlateJoint[40:,:,:]
valid_series_RW = RopewayJoint[40:,:,:]

train_BallLiftClass = np.zeros((40,200)).astype(np.int8)
train_BallLiftClass.fill(0)
train_BallRollClass = np.zeros((40,200)).astype(np.int8)
train_BallRollClass.fill(1)
train_BellRingLClass = np.zeros((40,200)).astype(np.int8)
train_BellRingLClass.fill(2)
train_BellRingRClass = np.zeros((40,200)).astype(np.int8)
train_BellRingRClass.fill(3)
train_BallRollPlateClass = np.zeros((40,200)).astype(np.int8)
train_BallRollPlateClass.fill(4)
train_RopewayClass = np.zeros((40,200)).astype(np.int8)
train_RopewayClass.fill(5)

valid_BallLiftClass = np.zeros((10,200)).astype(np.int8)
valid_BallLiftClass.fill(0)
valid_BallRollClass = np.zeros((10,200)).astype(np.int8)
valid_BallRollClass.fill(1)
valid_BellRingLClass = np.zeros((10,200)).astype(np.int8)
valid_BellRingLClass.fill(2)
valid_BellRingRClass = np.zeros((10,200)).astype(np.int8)
valid_BellRingRClass.fill(3)
valid_BallRollPlateClass = np.zeros((10,200)).astype(np.int8)
valid_BallRollPlateClass.fill(4)
valid_RopewayClass = np.zeros((10,200)).astype(np.int8)
valid_RopewayClass.fill(5)

train_series = np.vstack((train_series_BL, train_series_BR, train_series_BRL, train_series_BRR, train_series_BRP, train_series_RW))
valid_series = np.vstack((valid_series_BL, valid_series_BR, valid_series_BRL, valid_series_BRR, valid_series_BRP, valid_series_RW))

train_class = np.vstack((train_BallLiftClass, train_BallRollClass, train_BellRingLClass, train_BellRingRClass, train_BallRollPlateClass, train_RopewayClass))
valid_class = np.vstack((valid_BallLiftClass, valid_BallRollClass, valid_BellRingLClass, valid_BellRingRClass, valid_BallRollPlateClass, valid_RopewayClass))


print train_series.shape
print train_class.shape

print valid_series.shape
print valid_class.shape

# def layer(n):
#     return dict(form='bidirectional', worker='lstm', size=n)

def layer(n):
    return dict(worker='rnn', size=n)
 
e = theanets.Experiment(
    theanets.recurrent.Classifier,
    layers=(10, layer(20), 6),
    recurrent_error_start=0,
    batch_size=20
)

# trn_loss=[]
# valid_loss=[]
# i=0
# for train, valid in e.itertrain([train_series, train_class], [valid_series, valid_class], optimize='rmsprop', patience=10, validate_every=5,
#         max_gradient_norm=10,):
#     print('training loss:', train['loss'])
#     print('most recent validation loss:', valid['loss'])
#     trn_loss.append(train['loss'])
#     valid_loss.append(valid['loss'])
#                 
#                 
# np.savetxt('trn_loss.txt', trn_loss)
# np.savetxt('valid_loss.txt', valid_loss)
# plotLearningCurve()
# e.save("rnn-joint-classifier.gz") 

e.load("rnn-joint-classifier.gz")
print e.network.classify(RopewayJoint[41:42,:200,:])[:,:]
        
 
