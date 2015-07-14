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

trndata = ClassificationDataSet(10,1, nb_classes=6)
tstdata = ClassificationDataSet(10,1, nb_classes=6)

for i in range(6000):
    trndata.appendLinked(BallLiftJoint[i,:], [0])
for i in range(6000):
    trndata.appendLinked(BallRollJoint[i,:], [1])
for i in range(6000):
    trndata.appendLinked(BellRingLJoint[i,:], [2])
for i in range(6000):
    trndata.appendLinked(BellRingRJoint[i,:], [3])
for i in range(6000):
    trndata.appendLinked(BallRollPlateJoint[i,:], [4])
for i in range(6000):
    trndata.appendLinked(RopewayJoint[i,:], [5])
    
for i in range(6000,8000):
    tstdata.appendLinked(BallLiftJoint[i,:], [0])
for i in range(6000,8000):
    tstdata.appendLinked(BallRollJoint[i,:], [1])
for i in range(6000,8000):
    tstdata.appendLinked(BellRingLJoint[i,:], [2])
for i in range(6000,8000):
    tstdata.appendLinked(BellRingRJoint[i,:], [3])
for i in range(6000,8000):
    tstdata.appendLinked(BallRollPlateJoint[i,:], [4])
for i in range(6000,8000):
    tstdata.appendLinked(RopewayJoint[i,:], [5])
    
    
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()    
print 'Loaded Dataset!'

#####################
#####################
     
print 'Building Network'
MLPClassificationNet = buildNetwork(trndata.indim,153,trndata.outdim, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer) 

print "Number of weights:",MLPClassificationNet.paramdim

trainer = RPropMinusTrainer(MLPClassificationNet, dataset=trndata, verbose=True, weightdecay=0.01)
    
tstErrorCount=0
oldtstError=0
trn_error=[]
tst_error=[]
trn_class_accu=[]
tst_class_accu=[]
        
trnErrorPath='153sigmoid/trn_error'
tstErrorPath='153sigmoid/tst_error'
trnClassErrorPath='153sigmoid/trn_ClassAccu'
tstClassErrorPath='153sigmoid/tst_ClassAccu'
networkPath='153sigmoid/TrainUntilConv.xml'
figPath='153sigmoid/ErrorGraph'
 
#####################
#####################
# print "Training Data Length: ", len(trndata)
# print "Validation Data Length: ", len(tstdata)
#  
#                   
# print 'Start Training'
# time_start = time.time()
# while (tstErrorCount<100):
#     print "********** Classification with 153sigmoid with RP- **********"   
#     trnError=trainer.train()
#     tstError = trainer.testOnData(dataset=tstdata)
#     trnAccu = 100-percentError(trainer.testOnClassData(), trndata['class'])
#     tstAccu = 100-percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
#     trn_class_accu.append(trnAccu)
#     tst_class_accu.append(tstAccu)
#     trn_error.append(trnError)
#     tst_error.append(tstError)
#                                                                                                                                             
#     np.savetxt(trnErrorPath, trn_error)
#     np.savetxt(tstErrorPath, tst_error)
#     np.savetxt(trnClassErrorPath, trn_class_accu)
#     np.savetxt(tstClassErrorPath, tst_class_accu)
#                                                                                                                                           
#     if(oldtstError==0):
#         oldtstError = tstError
#                                                                                                                                               
#     if(oldtstError<tstError):
#         tstErrorCount = tstErrorCount+1
#         print 'No Improvement, count=%d' % tstErrorCount
#         print '    Old Validation Error:', oldtstError 
#         print 'Current Validation Error:', tstError
#                                                                                                                                               
#     if(oldtstError>tstError):
#         print 'Improvement made!'
#         print '    Old Validation Error:', oldtstError 
#         print 'Current Validation Error:', tstError
#         tstErrorCount=0
#         oldtstError = tstError
#         NetworkWriter.writeToFile(MLPClassificationNet, networkPath)
#         plotLearningCurve()
#      
#     
# trainingTime = time.time()-time_start
# trainingTime=np.reshape(trainingTime, (1))
# np.savetxt("153sigmoid/Trainingtime.txt", trainingTime)

####################
# Manual OFFLINE Test
####################        
MLPClassificationNet = NetworkReader.readFrom('153sigmoid//TrainUntilConv.xml')
trainer = RPropMinusTrainer(MLPClassificationNet, dataset=trndata, verbose=True, weightdecay=0.01)
print 'Loaded Trained Network!'
from random import randint

testingdata = ClassificationDataSet(10,1, nb_classes=6)

for i in range(8000,10000):
    testingdata.appendLinked(BallLiftJoint[i,:], [0])
for i in range(8000,10000):
    testingdata.appendLinked(BallRollJoint[i,:], [1])
for i in range(8000,10000):
    testingdata.appendLinked(BellRingLJoint[i,:], [2])
for i in range(8000,10000):
    testingdata.appendLinked(BellRingRJoint[i,:], [3])
for i in range(8000,10000):
    testingdata.appendLinked(BallRollPlateJoint[i,:], [4])
for i in range(8000,10000):
    testingdata.appendLinked(RopewayJoint[i,:], [5])

testingdata._convertToOneOfMany()

print MLPClassificationNet.paramdim
testingAccu = 100-percentError(trainer.testOnClassData(dataset=testingdata), testingdata['class'])
print testingAccu
# MLPClassificationNet = NetworkReader.readFrom('153sigmoid//TrainUntilConv.xml')
# print 'Loaded Trained Network!'
# from random import randint
# 
# print MLPClassificationNet.paramdim
# 
# 
# x = MLPClassificationNet.activate(RopewayJoint[10])
# print argmax(x)

