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

BallLiftJoint = BallLiftJoint.reshape((200,50,10))
BallRollJoint = BallRollJoint.reshape((200,50,10))
BellRingLJoint = BellRingLJoint.reshape((200,50,10))
BellRingRJoint = BellRingRJoint.reshape((200,50,10))
BallRollPlateJoint = BallRollPlateJoint.reshape((200,50,10))
RopewayJoint = RopewayJoint.reshape((200,50,10))

# plt.subplot(122)
# plt.plot(BallLiftJoint[1,:,:])
# plt.show()

train_series_BL = BallLiftJoint[:,:40,:]
train_series_BR = BallRollJoint[:,:40,:]
train_series_BRL = BellRingLJoint[:,:40,:]
train_series_BRR = BellRingRJoint[:,:40,:]
train_series_BRP = BallRollPlateJoint[:,:40,:]
train_series_RW = RopewayJoint[:,:40,:]

valid_series_BL = BallLiftJoint[:,40:,:]
valid_series_BR = BallRollJoint[:,40:,:]
valid_series_BRL = BellRingLJoint[:,40:,:]
valid_series_BRR = BellRingRJoint[:,40:,:]
valid_series_BRP = BallRollPlateJoint[:,40:,:]
valid_series_RW = RopewayJoint[:,40:,:]

train_BallLiftClass = np.zeros((200,40)).astype(np.int8)
train_BallLiftClass.fill(0)
train_BallRollClass = np.zeros((200,40)).astype(np.int8)
train_BallRollClass.fill(1)
train_BellRingLClass = np.zeros((200,40)).astype(np.int8)
train_BellRingLClass.fill(2)
train_BellRingRClass = np.zeros((200,40)).astype(np.int8)
train_BellRingRClass.fill(3)
train_BallRollPlateClass = np.zeros((200,40)).astype(np.int8)
train_BallRollPlateClass.fill(4)
train_RopewayClass = np.zeros((200,40)).astype(np.int8)
train_RopewayClass.fill(5)

valid_BallLiftClass = np.zeros((200,10)).astype(np.int8)
valid_BallLiftClass.fill(0)
valid_BallRollClass = np.zeros((200,10)).astype(np.int8)
valid_BallRollClass.fill(1)
valid_BellRingLClass = np.zeros((200,10)).astype(np.int8)
valid_BellRingLClass.fill(2)
valid_BellRingRClass = np.zeros((200,10)).astype(np.int8)
valid_BellRingRClass.fill(3)
valid_BallRollPlateClass = np.zeros((200,10)).astype(np.int8)
valid_BallRollPlateClass.fill(4)
valid_RopewayClass = np.zeros((200,10)).astype(np.int8)
valid_RopewayClass.fill(5)

train_series = np.hstack((train_series_BL, train_series_BR, train_series_BRL, train_series_BRR, train_series_BRP, train_series_RW))
valid_series = np.hstack((valid_series_BL, valid_series_BR, valid_series_BRL, valid_series_BRR, valid_series_BRP, valid_series_RW))

train_class = np.hstack((train_BallLiftClass, train_BallRollClass, train_BellRingLClass, train_BellRingRClass, train_BallRollPlateClass, train_RopewayClass))
valid_class = np.hstack((valid_BallLiftClass, valid_BallRollClass, valid_BellRingLClass, valid_BellRingRClass, valid_BallRollPlateClass, valid_RopewayClass))


print train_series.shape
print train_class.shape

print valid_series.shape
print valid_class.shape

def layer(n):
    return dict(form='bidirectional', worker='lstm', size=n)

# def layer(n):
#     return dict(worker='lstm', size=n)
 
e = theanets.Experiment(
    theanets.recurrent.Classifier,
    layers=(10, layer(20), layer(20), 6),
    recurrent_error_start=0,
    batch_size=20
)

 
# trn_loss=[]
# valid_loss=[]
# i=0
# for train, valid in e.itertrain([train_series, train_class], [valid_series, valid_class], optimize='rmsprop', patience=10, validate_every=5, input_noise=0.6,
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
# e.save("unilstm-joint-classifier.gz") 
        
        
BallLiftJoint = BallLiftJoint.reshape((10000,10))
fig=plt.figure(0)
plt.subplot(121)
plt.plot(BallLiftJoint[:200,:])


testdata = BallLiftJoint[:200,:]
testdata = testdata.reshape((200,1,10))

# plt.subplot(122)
# plt.plot(testdata[2,:,:])
# plt.show()


e.load("unilstm-joint-classifier.gz")
print e.network.classify(testdata)
# print e.network.classify(valid_series[:20,55:60,:])
# print e.network.classify(valid_series[:20,45:50,:])
# print e.network.classify(valid_series[:20,35:40,:])
# print e.network.classify(valid_series[:20,25:30,:])
# print e.network.classify(valid_series[:20,15:20,:])
# print e.network.classify(valid_series[:20,5:10,:])


