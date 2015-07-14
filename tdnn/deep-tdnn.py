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
from utils import load_mnist

def plotLearningCurve():
    fig=plt.figure(0, figsize=(10,8) )
    fig.clf()
    plt.ioff()
#     plt.subplot(211)
    plt.plot(trn_loss, label='Training Set Error', linestyle="--", linewidth=2)
    plt.plot(valid_loss, label='Validation Set Error', linewidth=2)
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
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

BL = BallLiftJoint.flatten()
BR = BallRollJoint.flatten()
BRL = BellRingLJoint.flatten()
BRR = BellRingRJoint.flatten()
BRP = BallRollPlateJoint.flatten()
RW = RopewayJoint.flatten()


BL = BL.reshape((500,200))
BR = BR.reshape((500,200))
BRL = BRL.reshape((500,200))
BRR = BRR.reshape((500,200))
BRP = BRP.reshape((500,200))
RW = RW.reshape((500,200))

train = np.vstack((BL[:400,:], BR[:400,:], BRL[:400,:], BRR[:400,:], BRP[:400,:], RW[:400,:]))
valid = np.vstack((BL[400:,:], BR[400:,:], BRL[400:,:], BRR[400:,:], BRP[400:,:], RW[400:,:]))

bl_train_class = np.zeros((400,1)).astype(np.int8)
br_train_class = np.zeros((400,1)).astype(np.int8)
brl_train_class = np.zeros((400,1)).astype(np.int8)
brr_train_class = np.zeros((400,1)).astype(np.int8)
brp_train_class = np.zeros((400,1)).astype(np.int8)
rw_train_class = np.zeros((400,1)).astype(np.int8)

bl_valid_class = np.zeros((100,1)).astype(np.int8)
br_valid_class = np.zeros((100,1)).astype(np.int8)
brl_valid_class = np.zeros((100,1)).astype(np.int8)
brr_valid_class = np.zeros((100,1)).astype(np.int8)
brp_valid_class = np.zeros((100,1)).astype(np.int8)
rw_valid_class = np.zeros((100,1)).astype(np.int8)

bl_train_class.fill(0)
br_train_class.fill(1)
brl_train_class.fill(2)
brr_train_class.fill(3)
brp_train_class.fill(4)
rw_train_class.fill(5)

bl_valid_class.fill(0)
br_valid_class.fill(1)
brl_valid_class.fill(2)
brr_valid_class.fill(3)
brp_valid_class.fill(4)
rw_valid_class.fill(5)

train_labels = np.vstack((bl_train_class, br_train_class, brl_train_class, brr_train_class, brp_train_class, rw_train_class))
valid_labels = np.vstack((bl_valid_class, br_valid_class, brl_valid_class, brr_valid_class, brp_valid_class, rw_valid_class))

train_labels = train_labels.reshape(2400)
valid_labels = valid_labels.reshape(600)

print train.shape
print valid.shape
print train_labels.shape
print valid_labels.shape


# mnist_train, mnist_valid, test = load_mnist(labels=True)
# print mnist_train[0].shape
# print mnist_train[1].shape





e = theanets.Experiment(
    theanets.Classifier,
    layers=(200, 300, 100, 64, 6),
    train_batches=1,
)
# e.train(train, valid, optimize='pretrain', patience=1, min_improvement=0.1)
# 
# 
# trn_loss=[]
# valid_loss=[]
# figPath = "loss_graph"
# for train, valid in e.itertrain([train, train_labels],[valid, valid_labels], patience=100):
#     print('training loss:', train['loss'])
#     print('most recent validation loss:', valid['loss'])
#     trn_loss.append(train['loss'])
#     valid_loss.append(valid['loss'])
# plotLearningCurve()
# e.save('tdnn.tar.gz')
#        
# np.savetxt('trn_loss.txt', trn_loss)
# np.savetxt('valid_loss.txt', valid_loss)


e.load('tdnn.tar.gz')

print e.network.classify(train[2000:2001, :])