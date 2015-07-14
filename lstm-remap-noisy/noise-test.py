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
from random import randint
from scipy.interpolate import interp1d
BallLiftJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BallLift/JointData.txt').astype(np.float32)
BallRollJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BallRoll/JointData.txt').astype(np.float32)
BellRingLJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BellRingL/JointData.txt').astype(np.float32)
BellRingRJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BellRingR/JointData.txt').astype(np.float32)
BallRollPlateJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BallRollPlate/JointData.txt').astype(np.float32)
RopewayJoint = np.loadtxt('../../20fpsFullBehaviorSampling/Ropeway/JointData.txt').astype(np.float32)

jointRemap = interp1d([-2.2,2.2],[-1,1])
BallLiftJoint = jointRemap(BallLiftJoint)
BallRollJoint = jointRemap(BallRollJoint)
BellRingLJoint = jointRemap(BellRingLJoint)
BellRingRJoint = jointRemap(BellRingRJoint)
BallRollPlateJoint = jointRemap(BallRollPlateJoint)
RopewayJoint = jointRemap(RopewayJoint)

LSTMClassificationNet = NetworkReader.readFrom('20LSTMCell/TrainUntilConv.xml')
print 'Loaded 20 LSTM Trained Network!'

twentylstmaccdata = []
twentylstmstddata = []
twentylstmstderror = []

predictedBLLabels = []
predictedBRLabels = []
predictedBRLLabels = []
predictedBRRLabels = []
predictedBRPLabels = []
predictedRWLabels = []

offset = 100
accuracyOverall = []
for testnumber in range(30):    
    start = randint(7999,9799)
    end = start + offset
    timestep=range(start,end)
    LSTMClassificationNet.reset()
    for i in timestep:
        x = LSTMClassificationNet.activate(BallLiftJoint[i])
    predictedBLLabels.append(argmax(x))
     
    start = randint(7999,9799)
    end = start + offset
    timestep=range(start,end)
    LSTMClassificationNet.reset()
    for i in timestep:
        x = LSTMClassificationNet.activate(BallRollJoint[i])
    predictedBRLabels.append(argmax(x))
     
    start = randint(7999,9799)
    end = start + offset
    timestep=range(start,end)
    LSTMClassificationNet.reset()
    for i in timestep:
        x = LSTMClassificationNet.activate(BellRingLJoint[i])
    predictedBRLLabels.append(argmax(x))
     
    start = randint(7999,9799)
    end = start + offset
    timestep=range(start,end)
    LSTMClassificationNet.reset()
    for i in timestep:
        x = LSTMClassificationNet.activate(BellRingRJoint[i])
    predictedBRRLabels.append(argmax(x))
     
    start = randint(7999,9799)
    end = start + offset
    timestep=range(start,end)
    LSTMClassificationNet.reset()
    for i in timestep:
        x = LSTMClassificationNet.activate(BallRollPlateJoint[i])
    predictedBRPLabels.append(argmax(x))
     
    start = randint(7999,9799)
    end = start + offset
    timestep=range(start,end)
    LSTMClassificationNet.reset()
    for i in timestep:
        x = LSTMClassificationNet.activate(RopewayJoint[i])
    predictedRWLabels.append(argmax(x))
     
     
testnumAcc = []
behaviorAccuracyfortestnumber = []
for testnumber in range(30):
    BLAcc = 100-percentError(predictedBLLabels[testnumber], [0])
    BRAcc = 100-percentError(predictedBRLabels[testnumber], [1])
    BRLAcc = 100-percentError(predictedBRLLabels[testnumber], [2])
    BRRAcc = 100-percentError(predictedBRRLabels[testnumber], [3])
    BRPAcc = 100-percentError(predictedBRPLabels[testnumber], [4])
    RWAcc = 100-percentError(predictedRWLabels[testnumber], [5])
     
    behaviorAccuracyfortestnumber.append((BLAcc + BRAcc + BRLAcc + BRRAcc + BRPAcc + RWAcc) / 6)
     
print behaviorAccuracyfortestnumber
 
print "Mean Accuracy for 30 trials:", np.mean(np.array(behaviorAccuracyfortestnumber))
print "Std Deviation for 30 trials:", np.std(np.array(behaviorAccuracyfortestnumber))

twentylstmaccdata.append(np.mean(np.array(behaviorAccuracyfortestnumber)))
twentylstmstddata.append(np.std(np.array(behaviorAccuracyfortestnumber)))

print len(twentylstmaccdata)

######## with noise ######
std_deviation = 0
mean = 0
while (std_deviation<=2.0):
    std_deviation += 0.1
    print "Std deviation:",std_deviation
    predictedBLLabels = []
    predictedBRLabels = []
    predictedBRLLabels = []
    predictedBRRLabels = []
    predictedBRPLabels = []
    predictedRWLabels = []
     
     
    offset = 100
    accuracyOverall = []
    for testnumber in range(30):
        BallLiftJoint = BallLiftJoint + np.random.normal(mean,std_deviation,(10000,10))
        BallRollJoint = BallRollJoint + np.random.normal(mean,std_deviation,(10000,10))
        BellRingLJoint = BellRingLJoint + np.random.normal(mean,std_deviation,(10000,10))
        BellRingRJoint = BellRingRJoint + np.random.normal(mean,std_deviation,(10000,10))
        BallRollPlateJoint = BallRollPlateJoint + np.random.normal(mean,std_deviation,(10000,10))
        RopewayJoint = RopewayJoint + np.random.normal(mean,std_deviation,(10000,10))
        
        start = randint(7999,9799)
        end = start + offset
        timestep=range(start,end)
        LSTMClassificationNet.reset()
        for i in timestep:
            x = LSTMClassificationNet.activate(BallLiftJoint[i])
        predictedBLLabels.append(argmax(x))
        
        start = randint(7999,9799)
        end = start + offset
        timestep=range(start,end)
        LSTMClassificationNet.reset()
        for i in timestep:
            x = LSTMClassificationNet.activate(BallRollJoint[i])
        predictedBRLabels.append(argmax(x))
        
        start = randint(7999,9799)
        end = start + offset
        timestep=range(start,end)
        LSTMClassificationNet.reset()
        for i in timestep:
            x = LSTMClassificationNet.activate(BellRingLJoint[i])
        predictedBRLLabels.append(argmax(x))
        
        start = randint(7999,9799)
        end = start + offset
        timestep=range(start,end)
        LSTMClassificationNet.reset()
        for i in timestep:
            x = LSTMClassificationNet.activate(BellRingRJoint[i])
        predictedBRRLabels.append(argmax(x))
        
        start = randint(7999,9799)
        end = start + offset
        timestep=range(start,end)
        LSTMClassificationNet.reset()
        for i in timestep:
            x = LSTMClassificationNet.activate(BallRollPlateJoint[i])
        predictedBRPLabels.append(argmax(x))
        
        start = randint(7999,9799)
        end = start + offset
        timestep=range(start,end)
        LSTMClassificationNet.reset()
        for i in timestep:
            x = LSTMClassificationNet.activate(RopewayJoint[i])
        predictedRWLabels.append(argmax(x))
        
        
    testnumAcc = []
    behaviorAccuracyfortestnumber = []
    for testnumber in range(30):
        BLAcc = 100-percentError(predictedBLLabels[testnumber], [0])
        BRAcc = 100-percentError(predictedBRLabels[testnumber], [1])
        BRLAcc = 100-percentError(predictedBRLLabels[testnumber], [2])
        BRRAcc = 100-percentError(predictedBRRLabels[testnumber], [3])
        BRPAcc = 100-percentError(predictedBRPLabels[testnumber], [4])
        RWAcc = 100-percentError(predictedRWLabels[testnumber], [5])
        
        behaviorAccuracyfortestnumber.append((BLAcc + BRAcc + BRLAcc + BRRAcc + BRPAcc + RWAcc) / 6)
        
#     print behaviorAccuracyfortestnumber

    print "Mean Accuracy for 30 trials:", np.mean(np.array(behaviorAccuracyfortestnumber))
    print "Std Deviation for 30 trials:", np.std(np.array(behaviorAccuracyfortestnumber))
    
    twentylstmaccdata.append(np.mean(np.array(behaviorAccuracyfortestnumber)))
    twentylstmstddata.append(np.std(np.array(behaviorAccuracyfortestnumber)))
    
    print len(twentylstmaccdata)
    

print twentylstmaccdata
print twentylstmstddata

for i in range(21):
    twentylstmstderror.append(twentylstmstddata[i]/np.sqrt(30))
    
print twentylstmstderror

np.savetxt("AccuracyData.txt",twentylstmaccdata )
np.savetxt("SigmaData.txt",twentylstmstddata )
np.savetxt("ErrorBarData.txt",twentylstmstderror )

    

plt.errorbar(y=twentylstmaccdata, x=np.arange(0.0,2.1,0.1), yerr=twentylstmstderror, label="20 LSTM", linewidth=2)
plt.xlim([0.0,2.1])
plt.xlabel(r"$\sigma$")
plt.ylabel("Classification Accuracy (%)")
plt.grid()
plt.legend()
plt.show()