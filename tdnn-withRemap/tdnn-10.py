import sys
import numpy as np
from pybrain.datasets import SequenceClassificationDataSet, ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain import LinearLayer, FullConnection, LSTMLayer, BiasUnit, MDLSTMLayer, IdentityConnection, TanhLayer, SoftmaxLayer, SigmoidLayer
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
# robotIP="192.168.0.108"
# tts=ALProxy("ALTextToSpeech", robotIP, 9559)
# motion = ALProxy("ALMotion", robotIP, 9559)
# memory = ALProxy("ALMemory", robotIP, 9559)
# posture = ALProxy("ALRobotPosture", robotIP, 9559)
# camProxy = ALProxy("ALVideoDevice", robotIP, 9559)
# resolution = 0    # kQQVGA
# colorSpace = 11   # RGB


# tts.say("Hello")
# posture.goToPosture("Crouch", 1.0)
# motion.rest()
#############
# Functions #
#############
   
def plotLearningCurve():
    fig=plt.figure(0, figsize=(10,8) )
    fig.clf()
    plt.ioff()
    plt.subplot(211)
    plt.plot(trn_error, label='Training Set Error', linestyle="--", linewidth=2)
    plt.plot(tst_error, label='Validation Set Error', linewidth=2)
    plt.title('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
       
    plt.subplot(212)
    plt.plot(trn_class_accu, label='Training Set Accuracy', linestyle="--", linewidth=2)
    plt.plot(tst_class_accu, label='Validation Set Accuracy', linewidth=2)
    plt.ylim([0,103])
    plt.ylabel('Percent')
    plt.xlabel('Epoch')
    plt.title('Classification Accuracy')
    plt.legend(loc=4)
       
#     plt.draw()
    plt.tight_layout(pad=2.1)
    plt.savefig(figPath)
           
################
# Load Dataset #
################

# Original Joint data
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


trndata = ClassificationDataSet(100,1, nb_classes=6)
tstdata = ClassificationDataSet(100,1, nb_classes=6)

BL = BallLiftJoint.flatten()
BR = BallRollJoint.flatten()
BRL = BellRingLJoint.flatten()
BRR = BellRingRJoint.flatten()
BRP = BallRollPlateJoint.flatten()
RW = RopewayJoint.flatten()

for i in range(0,60000, 100):
    trndata.addSample(BL[i:i+100], [0])
    trndata.addSample(BR[i:i+100], [1])
    trndata.addSample(BRL[i:i+100], [2])
    trndata.addSample(BRR[i:i+100], [3])
    trndata.addSample(BRP[i:i+100], [4])
    trndata.addSample(RW[i:i+100], [5])
    
for i in range(60000,80000, 100):
    tstdata.addSample(BL[i:i+100], [0])
    tstdata.addSample(BR[i:i+100], [1])
    tstdata.addSample(BRL[i:i+100], [2])
    tstdata.addSample(BRR[i:i+100], [3])
    tstdata.addSample(BRP[i:i+100], [4])
    tstdata.addSample(RW[i:i+100], [5])

# trndata.addSample(BL[:500], [0])
# trndata.addSample(BL[500:1000], [0])
# trndata.addSample(BL[1000:1500], [0])
# trndata.addSample(BL[1500:2000], [0])
# trndata.addSample(BL[2000:2500], [0])
# trndata.addSample(BL[2500:3000], [0])
# trndata.addSample(BL[3000:3500], [0])
# trndata.addSample(BL[3500:4000], [0])
# trndata.addSample(BL[4000:4500], [0])
# trndata.addSample(BL[4500:5000], [0])
# trndata.addSample(BL[5000:5500], [0])
# trndata.addSample(BL[5500:6000], [0])
# 
# trndata.addSample(BR[:500], [1])
# trndata.addSample(BR[500:1000], [1])
# trndata.addSample(BR[1000:1500], [1])
# trndata.addSample(BR[1500:2000], [1])
# trndata.addSample(BR[2000:2500], [1])
# trndata.addSample(BR[2500:3000], [1])
# trndata.addSample(BR[3000:3500], [1])
# trndata.addSample(BR[3500:4000], [1])
# trndata.addSample(BR[4000:4500], [1])
# trndata.addSample(BR[4500:5000], [1])
# trndata.addSample(BR[5000:5500], [1])
# trndata.addSample(BR[5500:6000], [1])
# 
# trndata.addSample(BRL[:500], [2])
# trndata.addSample(BRL[500:1000], [2])
# trndata.addSample(BRL[1000:1500], [2])
# trndata.addSample(BRL[1500:2000], [2])
# trndata.addSample(BRL[2000:2500], [2])
# trndata.addSample(BRL[2500:3000], [2])
# trndata.addSample(BRL[3000:3500], [2])
# trndata.addSample(BRL[3500:4000], [2])
# trndata.addSample(BRL[4000:4500], [2])
# trndata.addSample(BRL[4500:5000], [2])
# trndata.addSample(BRL[5000:5500], [2])
# trndata.addSample(BRL[5500:6000], [2])
# 
# trndata.addSample(BRR[:500], [3])
# trndata.addSample(BRR[500:1000], [3])
# trndata.addSample(BRR[1000:1500], [3])
# trndata.addSample(BRR[1500:2000], [3])
# trndata.addSample(BRR[2000:2500], [3])
# trndata.addSample(BRR[2500:3000], [3])
# trndata.addSample(BRR[3000:3500], [3])
# trndata.addSample(BRR[3500:4000], [3])
# trndata.addSample(BRR[4000:4500], [3])
# trndata.addSample(BRR[4500:5000], [3])
# trndata.addSample(BRR[5000:5500], [3])
# trndata.addSample(BRR[5500:6000], [3])
# 
# trndata.addSample(BRP[:500], [4])
# trndata.addSample(BRP[500:1000], [4])
# trndata.addSample(BRP[1000:1500], [4])
# trndata.addSample(BRP[1500:2000], [4])
# trndata.addSample(BRP[2000:2500], [4])
# trndata.addSample(BRP[2500:3000], [4])
# trndata.addSample(BRP[3000:3500], [4])
# trndata.addSample(BRP[3500:4000], [4])
# trndata.addSample(BRP[4000:4500], [4])
# trndata.addSample(BRP[4500:5000], [4])
# trndata.addSample(BRP[5000:5500], [4])
# trndata.addSample(BRP[5500:6000], [4])
# 
# trndata.addSample(RW[:500], [5])
# trndata.addSample(RW[500:1000], [5])
# trndata.addSample(RW[1000:1500], [5])
# trndata.addSample(RW[1500:2000], [5])
# trndata.addSample(RW[2000:2500], [5])
# trndata.addSample(RW[2500:3000], [5])
# trndata.addSample(RW[3000:3500], [5])
# trndata.addSample(RW[3500:4000], [5])
# trndata.addSample(RW[4000:4500], [5])
# trndata.addSample(RW[4500:5000], [5])
# trndata.addSample(RW[5000:5500], [5])
# trndata.addSample(RW[5500:6000], [5])
# 
# 
# 
# tstdata.addSample(BL[6000:6500], [0])
# tstdata.addSample(BL[6500:7000], [0])
# tstdata.addSample(BL[7000:7500], [0])
# tstdata.addSample(BL[7500:8000], [0])
# tstdata.addSample(BL[8000:8500], [0])
# tstdata.addSample(BL[8500:9000], [0])
# tstdata.addSample(BL[9000:9500], [0])
# tstdata.addSample(BL[9500:10000], [0])
# 
# tstdata.addSample(BR[6000:6500], [1])
# tstdata.addSample(BR[6500:7000], [1])
# tstdata.addSample(BR[7000:7500], [1])
# tstdata.addSample(BR[7500:8000], [1])
# tstdata.addSample(BR[8000:8500], [1])
# tstdata.addSample(BR[8500:9000], [1])
# tstdata.addSample(BR[9000:9500], [1])
# tstdata.addSample(BR[9500:10000], [1])
# 
# tstdata.addSample(BRL[6000:6500], [2])
# tstdata.addSample(BRL[6500:7000], [2])
# tstdata.addSample(BRL[7000:7500], [2])
# tstdata.addSample(BRL[7500:8000], [2])
# tstdata.addSample(BRL[8000:8500], [2])
# tstdata.addSample(BRL[8500:9000], [2])
# tstdata.addSample(BRL[9000:9500], [2])
# tstdata.addSample(BRL[9500:10000], [2])
# 
# tstdata.addSample(BRR[6000:6500], [3])
# tstdata.addSample(BRR[6500:7000], [3])
# tstdata.addSample(BRR[7000:7500], [3])
# tstdata.addSample(BRR[7500:8000], [3])
# tstdata.addSample(BRR[8000:8500], [3])
# tstdata.addSample(BRR[8500:9000], [3])
# tstdata.addSample(BRR[9000:9500], [3])
# tstdata.addSample(BRR[9500:10000], [3])
# 
# tstdata.addSample(BRP[6000:6500], [4])
# tstdata.addSample(BRP[6500:7000], [4])
# tstdata.addSample(BRP[7000:7500], [4])
# tstdata.addSample(BRP[7500:8000], [4])
# tstdata.addSample(BRP[8000:8500], [4])
# tstdata.addSample(BRP[8500:9000], [4])
# tstdata.addSample(BRP[9000:9500], [4])
# tstdata.addSample(BRP[9500:10000], [4])
# 
# tstdata.addSample(RW[6000:6500], [5])
# tstdata.addSample(RW[6500:7000], [5])
# tstdata.addSample(RW[7000:7500], [5])
# tstdata.addSample(RW[7500:8000], [5])
# tstdata.addSample(RW[8000:8500], [5])
# tstdata.addSample(RW[8500:9000], [5])
# tstdata.addSample(RW[9000:9500], [5])
# tstdata.addSample(RW[9500:10000], [5])

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()    
print 'Loaded Dataset!'
 
 
print 'Building Network'
TDNNClassificationNet = buildNetwork(trndata.indim,25,trndata.outdim, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer) 
 
print "Number of weights:",TDNNClassificationNet.paramdim
 
trainer = RPropMinusTrainer(TDNNClassificationNet, dataset=trndata, verbose=True, weightdecay=0.01)
 
print TDNNClassificationNet.paramdim
 
tstErrorCount=0
oldtstError=0
trn_error=[]
tst_error=[]
trn_class_accu=[]
tst_class_accu=[]
         
trnErrorPath='25sigmoid/trn_error'
tstErrorPath='25sigmoid/tst_error'
trnClassErrorPath='25sigmoid/trn_ClassAccu'
tstClassErrorPath='25sigmoid/tst_ClassAccu'
networkPath='25sigmoid/TrainUntilConv.xml'
figPath='25sigmoid/ErrorGraph'
  
#####################
#####################
print "Training Data Length: ", len(trndata)
print "Validation Data Length: ", len(tstdata)
      
                       
print 'Start Training'
time_start = time.time()
while (tstErrorCount<100):
    print "********** Classification with 25sigmoid with RP- **********"   
    trnError=trainer.train()
    tstError = trainer.testOnData(dataset=tstdata)
    trnAccu = 100-percentError(trainer.testOnClassData(), trndata['class'])
    tstAccu = 100-percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
    trn_class_accu.append(trnAccu)
    tst_class_accu.append(tstAccu)
    trn_error.append(trnError)
    tst_error.append(tstError)
                                                                                                                                                 
    np.savetxt(trnErrorPath, trn_error)
    np.savetxt(tstErrorPath, tst_error)
    np.savetxt(trnClassErrorPath, trn_class_accu)
    np.savetxt(tstClassErrorPath, tst_class_accu)
                                                                                                                                               
    if(oldtstError==0):
        oldtstError = tstError
                                                                                                                                                   
    if(oldtstError<tstError):
        tstErrorCount = tstErrorCount+1
        print 'No Improvement, count=%d' % tstErrorCount
        print '    Old Validation Error:', oldtstError 
        print 'Current Validation Error:', tstError
                                                                                                                                                   
    if(oldtstError>tstError):
        print 'Improvement made!'
        print '    Old Validation Error:', oldtstError 
        print 'Current Validation Error:', tstError
        tstErrorCount=0
        oldtstError = tstError
        NetworkWriter.writeToFile(TDNNClassificationNet, networkPath)
        plotLearningCurve()
          
         
trainingTime = time.time()-time_start
trainingTime=np.reshape(trainingTime, (1))
np.savetxt("25sigmoid/Trainingtime.txt", trainingTime)


####################
# Manual OFFLINE Test
####################        
# TDNNClassificationNet = NetworkReader.readFrom('25sigmoid/TrainUntilConv.xml')
# print 'Loaded Trained Network!'
#  
# print TDNNClassificationNet.paramdim
#  
# testdata = RopewayJoint.flatten()[90000:90100]
# x = TDNNClassificationNet.activate(testdata)
# print argmax(x)