import sys
import numpy as np
from pybrain.datasets import SequenceClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain import LinearLayer, FullConnection, LSTMLayer, BiasUnit, MDLSTMLayer, IdentityConnection, TanhLayer, SoftmaxLayer
from pybrain.utilities import percentError
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from naoqi import ALProxy
import Image
import time
import theanets
import vision_definitions
from numpy.random.mtrand import randint
from numpy import argmax
from scipy.interpolate import interp1d
networkPath='20LSTMCell/TrainUntilConv.xml'

def plotConfusionMatrixSequentialData():
    predictedBLLabels = []
    predictedBRLabels = []
    predictedBRLLabels = []
    predictedBRRLabels = []
    predictedBRPLabels = []
    predictedRWLabels = []
    
    # for all data in training set, get labels for each sequence in 100 timesteps
    LSTMClassificationNet.reset()
    for sequenceStart in range(0,6000,100): 
        timestep=range(sequenceStart,sequenceStart+100)
        for i in timestep:
            x = LSTMClassificationNet.activate(BallLiftJoint[i])
        predictedBLLabels.append(argmax(x))
    
    LSTMClassificationNet.reset()
    for sequenceStart in range(0,6000,100): 
        timestep=range(sequenceStart,sequenceStart+100)
        for i in timestep:
            x = LSTMClassificationNet.activate(BallRollJoint[i])
        predictedBRLabels.append(argmax(x))
    
    LSTMClassificationNet.reset()
    for sequenceStart in range(0,6000,100): 
        timestep=range(sequenceStart,sequenceStart+100)
        for i in timestep:
            x = LSTMClassificationNet.activate(BellRingLJoint[i])
        predictedBRLLabels.append(argmax(x))
        
    LSTMClassificationNet.reset()
    for sequenceStart in range(0,6000,100): 
        timestep=range(sequenceStart,sequenceStart+100)
        for i in timestep:
            x = LSTMClassificationNet.activate(BellRingRJoint[i])
        predictedBRRLabels.append(argmax(x))
        
    LSTMClassificationNet.reset()
    for sequenceStart in range(0,6000,100): 
        timestep=range(sequenceStart,sequenceStart+100)
        for i in timestep:
            x = LSTMClassificationNet.activate(BallRollPlateJoint[i])
        predictedBRPLabels.append(argmax(x))
        
    LSTMClassificationNet.reset()
    for sequenceStart in range(0,6000,100): 
        timestep=range(sequenceStart,sequenceStart+100)
        for i in timestep:
            x = LSTMClassificationNet.activate(RopewayJoint[i])
        predictedRWLabels.append(argmax(x))
    
    # print predictedBLLabels
    # print predictedBRLabels
    # print predictedBRLLabels
    # print predictedBRRLabels
    # print predictedBRPLabels
    # print predictedRWLabels
    
    
    truelabelzero = [0] * 60 # 60 coresspnds to number of sequnece in 6000 traiing data points
    truelabelone = [1] * 60
    truelabeltwo = [2] * 60
    truelabelthree = [3] * 60
    truelabelfour = [4] * 60
    truelabelfive = [5] * 60
    
    # combind all list into a bigger list
    predictedlabels = predictedBLLabels+predictedBRLabels+predictedBRLLabels+predictedBRRLabels+predictedBRPLabels+predictedRWLabels
    actuallabels = truelabelzero+truelabelone+truelabeltwo+truelabelthree+truelabelfour+truelabelfive
    
    
    # BLAcc = 100-percentError(predictedBLLabels, [0])
    # BRAcc = 100-percentError(predictedBRLabels, [1])
    # BRLAcc = 100-percentError(predictedBRLLabels, [2])
    # BRRAcc = 100-percentError(predictedBRRLabels, [3])
    # BRPAcc = 100-percentError(predictedBRPLabels, [4])
    # RWAcc = 100-percentError(predictedRWLabels, [5])
    # 
    # print BLAcc, BRAcc, BRLAcc, BRRAcc, BRPAcc, RWAcc
    
    
    cm = confusion_matrix(actuallabels, predictedlabels)
    
    tick_marks = np.arange(0,6)
    behavior_names = ['Ball Lift', 'Ball Roll', 'Bell Ring L', 'Bell Ring R', 'Ball Roll Plate', 'Ropeway']
    fig= plt.figure(2, figsize=(12,9))
    plt.clf()
    plt.title("Confusion matrix on the Training Set")
    
    # enable this piece of code to see values of pixel
    width = len(cm)
    height = len(cm[0])
    ax = fig.add_subplot(111)
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(tick_marks, behavior_names)
    plt.yticks(tick_marks, behavior_names)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    

# Original Joint data
BallLiftJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BallLift/JointData.txt').astype(np.float32)
BallRollJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BallRoll/JointData.txt').astype(np.float32)
BellRingLJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BellRingL/JointData.txt').astype(np.float32)
BellRingRJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BellRingR/JointData.txt').astype(np.float32)
BallRollPlateJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BallRollPlate/JointData.txt').astype(np.float32)
RopewayJoint = np.loadtxt('../../20fpsFullBehaviorSampling/Ropeway/JointData.txt').astype(np.float32)

trndata = SequenceClassificationDataSet(10,1, nb_classes=6, class_labels=["BL", "BR", "BRL", "BRR", "BRP", "RW"])
tstdata = SequenceClassificationDataSet(10,1, nb_classes=6, class_labels=["BL", "BR", "BRL", "BRR", "BRP", "RW"])
    
for i in range(6000):
    if i%100==0:
        trndata.newSequence()
    trndata.appendLinked(BallLiftJoint[i,:], [0])
for i in range(6000):
    if i%100==0:
        trndata.newSequence()
    trndata.appendLinked(BallRollJoint[i,:], [1])
for i in range(6000):
    if i%100==0:
        trndata.newSequence()
    trndata.appendLinked(BellRingLJoint[i,:], [2])
for i in range(6000):
    if i%100==0:
        trndata.newSequence()
    trndata.appendLinked(BellRingRJoint[i,:], [3])
for i in range(6000):
    if i%100==0:
        trndata.newSequence()
    trndata.appendLinked(BallRollPlateJoint[i,:], [4])
for i in range(6000):
    if i%100==0:
        trndata.newSequence()
    trndata.appendLinked(RopewayJoint[i,:], [5])
   
          
          
for i in range(6000,8000):
    if i%100==0:
        tstdata.newSequence()
    tstdata.appendLinked(BallLiftJoint[i,:], [0])
for i in range(6000,8000):
    if i%100==0:
        tstdata.newSequence()
    tstdata.appendLinked(BallRollJoint[i,:], [1])
for i in range(6000,8000):
    if i%100==0:
        tstdata.newSequence()
    tstdata.appendLinked(BellRingLJoint[i,:], [2])
for i in range(6000,8000):
    if i%100==0:
        tstdata.newSequence()
    tstdata.appendLinked(BellRingRJoint[i,:], [3])
for i in range(6000,8000):
    if i%100==0:
        tstdata.newSequence()
    tstdata.appendLinked(BallRollPlateJoint[i,:], [4])
for i in range(6000,8000):
    if i%100==0:
        tstdata.newSequence()
    tstdata.appendLinked(RopewayJoint[i,:], [5])
 
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()    
print 'Loaded Dataset!'


    
LSTMClassificationNet = NetworkReader.readFrom(networkPath)
trainer = RPropMinusTrainer(LSTMClassificationNet, dataset=trndata, verbose=True, weightdecay=0.01)
plotConfusionMatrixSequentialData()
plt.savefig("confmatrix")
plt.show()
