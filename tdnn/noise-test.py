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

tdnnclassifier = NetworkReader.readFrom('25sigmoid/TrainUntilConv.xml')
print 'Loaded 25 sigmoid TDNN Trained Network!'

twentylstmaccdata = []
twentylstmstddata = []
twentylstmstderror = []

predictedBLLabels = []
predictedBRLabels = []
predictedBRLLabels = []
predictedBRRLabels = []
predictedBRPLabels = []
predictedRWLabels = []
print "1st Iteration, noiseless test data"
offset = 100
accuracyOverall = []
for testnumber in range(30):    
    start = randint(8000,9980)
    x = tdnnclassifier.activate(BallLiftJoint[start:start+10].flatten())
    predictedBLLabels.append(argmax(x))
    
    start = randint(8000,9980)
    x = tdnnclassifier.activate(BallRollJoint[start:start+10].flatten())
    predictedBRLabels.append(argmax(x))
    
    start = randint(8000,9980)
    x = tdnnclassifier.activate(BellRingLJoint[start:start+10].flatten())
    predictedBRLLabels.append(argmax(x))
    
    start = randint(8000,9980)
    x = tdnnclassifier.activate(BellRingRJoint[start:start+10].flatten())
    predictedBRRLabels.append(argmax(x))
    
    start = randint(8000,9980)
    x = tdnnclassifier.activate(BallRollPlateJoint[start:start+10].flatten())
    predictedBRPLabels.append(argmax(x))
    
    start = randint(8000,9980)
    x = tdnnclassifier.activate(RopewayJoint[start:start+10].flatten())
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
 
print "Length of data (iteration number):",len(twentylstmaccdata)
 
# ######## with noise ######
std_deviation = 0
mean = 0
while (std_deviation<=2.0):
    std_deviation += 0.1
    print "Gaussian Noise std deviation:",std_deviation
    predictedBLLabels = []
    predictedBRLabels = []
    predictedBRLLabels = []
    predictedBRRLabels = []
    predictedBRPLabels = []
    predictedRWLabels = []
       
       
    offset = 100
    accuracyOverall = []
    for testnumber in range(30): # test for 30 times
        BallLiftJoint = BallLiftJoint + np.random.normal(mean,std_deviation,(10000,10))
        BallRollJoint = BallRollJoint + np.random.normal(mean,std_deviation,(10000,10))
        BellRingLJoint = BellRingLJoint + np.random.normal(mean,std_deviation,(10000,10))
        BellRingRJoint = BellRingRJoint + np.random.normal(mean,std_deviation,(10000,10))
        BallRollPlateJoint = BallRollPlateJoint + np.random.normal(mean,std_deviation,(10000,10))
        RopewayJoint = RopewayJoint + np.random.normal(mean,std_deviation,(10000,10))
          
        start = randint(8000,9980) # randomly select any data in this range
        x = tdnnclassifier.activate(BallLiftJoint[start:start+10].flatten())
        predictedBLLabels.append(argmax(x))
        
        start = randint(8000,9980)
        x = tdnnclassifier.activate(BallRollJoint[start:start+10].flatten())
        predictedBRLabels.append(argmax(x))
        
        start = randint(8000,9980)
        x = tdnnclassifier.activate(BellRingLJoint[start:start+10].flatten())
        predictedBRLLabels.append(argmax(x))
        
        start = randint(8000,9980)
        x = tdnnclassifier.activate(BellRingRJoint[start:start+10].flatten())
        predictedBRRLabels.append(argmax(x))
        
        start = randint(8000,9980)
        x = tdnnclassifier.activate(BallRollPlateJoint[start:start+10].flatten())
        predictedBRPLabels.append(argmax(x))
        
        start = randint(8000,9980)
        x = tdnnclassifier.activate(RopewayJoint[start:start+10].flatten())
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
      
    print "Length of data (iteration number)",len(twentylstmaccdata)
      
  
print "Accuracy:",twentylstmaccdata
print "Std Deviation:",twentylstmstddata
  
for i in range(21): # 21 because from sigma 0 to 2 in 0.1 steps we have 20 steps
    twentylstmstderror.append(twentylstmstddata[i]/np.sqrt(30))
      
print "Error bar",twentylstmstderror

np.savetxt("AccuracyData.txt",twentylstmaccdata )
np.savetxt("SigmaData.txt",twentylstmstddata )
np.savetxt("ErrorBarData.txt",twentylstmstderror )
      
  
plt.errorbar(y=twentylstmaccdata, x=np.arange(0.0,2.1,0.1), yerr=twentylstmstderror, label="25 Sigmoid TDNN", linewidth=2)
plt.xlim([0.0,2.1])
plt.xlabel(r"$\sigma$")
plt.ylabel("Classification Accuracy (%)")
plt.grid()
plt.legend()
plt.savefig('tdnnnoisetest')
plt.show()