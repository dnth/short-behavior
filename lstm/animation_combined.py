from matplotlib import animation
from naoqi import ALProxy
import matplotlib.pyplot as plt
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
import numpy as np
import time

robotIP="192.168.0.101"
tts=ALProxy("ALTextToSpeech", robotIP, 9559)
motion = ALProxy("ALMotion", robotIP, 9559)
memory = ALProxy("ALMemory", robotIP, 9559)
posture = ALProxy("ALRobotPosture", robotIP, 9559)
camProxy = ALProxy("ALVideoDevice", robotIP, 9559)
resolution = 0    # kQQVGA
colorSpace = 11   # RGB

###########################################
# For long behavior with 8 classes - BLIT #
###########################################
# First set up the figure, the axis, and the plot element we want to animate

fig = plt.figure(1,figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
ax = plt.axes(xlim=(0,800), ylim=(-0.5, 2))
plt.title("Probability Timeline")
plt.xlabel('Time Step')
plt.ylabel('Probability')


lineBL, = ax.plot([], [], lw=2, label='BL')
lineBR, = ax.plot([], [], lw=2, label='BR')
lineBRL, = ax.plot([], [], lw=2, label='BRL')
lineBRR, = ax.plot([], [], lw=2, label='BRR') 
lineBRP, = ax.plot([], [], lw=2, label='BRP')
lineRW, = ax.plot([], [], lw=2, label='RW')

plt.legend(loc='upper center', bbox_to_anchor=(0.81, 1.), fancybox=True, shadow=True, ncol=2)
plt.grid(True)


BL_list = [0] * 800
BR_list = [0] * 800
BRL_list = [0] * 800
BRR_list = [0] * 800
BRP_list = [0] * 800
RW_list = [0] * 800


LSTMClassificationNet = NetworkReader.readFrom("20LSTMCell/TrainUntilConv.xml")
print 'Loaded Trained Network!'
RShoulderPitchTestData = memory.getData("Device/SubDeviceList/RShoulderPitch/Position/Sensor/Value")
RShoulderRollTestData = memory.getData("Device/SubDeviceList/RShoulderRoll/Position/Sensor/Value")
RElbowRollTestData = memory.getData("Device/SubDeviceList/RElbowRoll/Position/Sensor/Value")
RElbowYawTestData = memory.getData("Device/SubDeviceList/RElbowYaw/Position/Sensor/Value")
RWristYawTestData = memory.getData("Device/SubDeviceList/RWristYaw/Position/Sensor/Value")
          
LShoulderPitchTestData = memory.getData("Device/SubDeviceList/LShoulderPitch/Position/Sensor/Value")
LShoulderRollTestData = memory.getData("Device/SubDeviceList/LShoulderRoll/Position/Sensor/Value")
LElbowRollTestData = memory.getData("Device/SubDeviceList/LElbowRoll/Position/Sensor/Value")
LElbowYawTestData = memory.getData("Device/SubDeviceList/LElbowYaw/Position/Sensor/Value")
LWristYawTestData = memory.getData("Device/SubDeviceList/LWristYaw/Position/Sensor/Value")

       
LSTMNet_output = LSTMClassificationNet.activate([RShoulderPitchTestData, RShoulderRollTestData, RElbowRollTestData, RElbowYawTestData, RWristYawTestData, 
                                               LShoulderPitchTestData, LShoulderRollTestData, LElbowRollTestData, LElbowYawTestData, LWristYawTestData])

# plot once and set initial height to zero
fig1 = plt.figure(2, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
rects = plt.bar(np.arange(6), LSTMNet_output, align='center')
plt.xticks(np.arange(8) , ('BL', 'BR', 'BRL', 'BRR', 'BRP','RW'))
plt.xlim([-0.5,6])
plt.ylim([0,1.1])
plt.title("Behavior Classification Histogram")
plt.ylabel('Probability')
plt.xlabel('Behavior')
plt.grid(True)
for rect, h in zip(rects, LSTMNet_output):
        rect.set_height(0)


def initLine():
    lineBL.set_data([], [])
    return lineBL,lineBR,lineBRL,lineBRR,lineBRP,lineRW,

# animation function.  This is called sequentially
def animateLine(i): 
    time_start = time.time()
    
    RShoulderPitchTestData = memory.getData("Device/SubDeviceList/RShoulderPitch/Position/Sensor/Value")
    RShoulderRollTestData = memory.getData("Device/SubDeviceList/RShoulderRoll/Position/Sensor/Value")
    RElbowRollTestData = memory.getData("Device/SubDeviceList/RElbowRoll/Position/Sensor/Value")
    RElbowYawTestData = memory.getData("Device/SubDeviceList/RElbowYaw/Position/Sensor/Value")
    RWristYawTestData = memory.getData("Device/SubDeviceList/RWristYaw/Position/Sensor/Value")
              
    LShoulderPitchTestData = memory.getData("Device/SubDeviceList/LShoulderPitch/Position/Sensor/Value")
    LShoulderRollTestData = memory.getData("Device/SubDeviceList/LShoulderRoll/Position/Sensor/Value")
    LElbowRollTestData = memory.getData("Device/SubDeviceList/LElbowRoll/Position/Sensor/Value")
    LElbowYawTestData = memory.getData("Device/SubDeviceList/LElbowYaw/Position/Sensor/Value")
    LWristYawTestData = memory.getData("Device/SubDeviceList/LWristYaw/Position/Sensor/Value")
    
           
    LSTMNet_output = LSTMClassificationNet.activate([RShoulderPitchTestData, RShoulderRollTestData, RElbowRollTestData, RElbowYawTestData, RWristYawTestData, 
                                               LShoulderPitchTestData, LShoulderRollTestData, LElbowRollTestData, LElbowYawTestData, LWristYawTestData])
    
    BL_outputNode,BR_outputNode,BRL_outputNode,BRR_outputNode,BRP_outputNode,RW_outputNode = LSTMNet_output
    
    BL_list.append(BL_outputNode)
    BR_list.append(BR_outputNode)
    BRL_list.append(BRL_outputNode)
    BRR_list.append(BRR_outputNode)
    BRP_list.append(BRP_outputNode)
    RW_list.append(RW_outputNode)
    
    
    del BL_list[0]
    del BR_list[0]
    del BRL_list[0]
    del BRR_list[0]
    del BRP_list[0]
    del RW_list[0]
    
    lineBL.set_data(np.arange(0,800,1),BL_list)
    lineBR.set_data(np.arange(0,800,1),BR_list)
    lineBRL.set_data(np.arange(0,800,1),BRL_list)
    lineBRR.set_data(np.arange(0,800,1),BRR_list)
    lineBRP.set_data(np.arange(0,800,1),BRP_list)
    lineRW.set_data(np.arange(0,800,1),RW_list)
    
    print "Line FPS:",1/(time.time()-time_start)
    return lineBL,lineBR,lineBRL,lineBRR,lineBRP,lineRW,


def initBar():
    global rects 
    RShoulderPitchTestData = memory.getData("Device/SubDeviceList/RShoulderPitch/Position/Sensor/Value")
    RShoulderRollTestData = memory.getData("Device/SubDeviceList/RShoulderRoll/Position/Sensor/Value")
    RElbowRollTestData = memory.getData("Device/SubDeviceList/RElbowRoll/Position/Sensor/Value")
    RElbowYawTestData = memory.getData("Device/SubDeviceList/RElbowYaw/Position/Sensor/Value")
    RWristYawTestData = memory.getData("Device/SubDeviceList/RWristYaw/Position/Sensor/Value")
              
    LShoulderPitchTestData = memory.getData("Device/SubDeviceList/LShoulderPitch/Position/Sensor/Value")
    LShoulderRollTestData = memory.getData("Device/SubDeviceList/LShoulderRoll/Position/Sensor/Value")
    LElbowRollTestData = memory.getData("Device/SubDeviceList/LElbowRoll/Position/Sensor/Value")
    LElbowYawTestData = memory.getData("Device/SubDeviceList/LElbowYaw/Position/Sensor/Value")
    LWristYawTestData = memory.getData("Device/SubDeviceList/LWristYaw/Position/Sensor/Value")
    
    LSTMNet_output = LSTMClassificationNet.activate([RShoulderPitchTestData, RShoulderRollTestData, RElbowRollTestData, RElbowYawTestData, RWristYawTestData, 
                                               LShoulderPitchTestData, LShoulderRollTestData, LElbowRollTestData, LElbowYawTestData, LWristYawTestData])
    
    return rects

# animation function.  This is called sequentially
def animateBar(i): 
    time_start = time.time()
    
    RShoulderPitchTestData = memory.getData("Device/SubDeviceList/RShoulderPitch/Position/Sensor/Value")
    RShoulderRollTestData = memory.getData("Device/SubDeviceList/RShoulderRoll/Position/Sensor/Value")
    RElbowRollTestData = memory.getData("Device/SubDeviceList/RElbowRoll/Position/Sensor/Value")
    RElbowYawTestData = memory.getData("Device/SubDeviceList/RElbowYaw/Position/Sensor/Value")
    RWristYawTestData = memory.getData("Device/SubDeviceList/RWristYaw/Position/Sensor/Value")     
    LShoulderPitchTestData = memory.getData("Device/SubDeviceList/LShoulderPitch/Position/Sensor/Value")
    LShoulderRollTestData = memory.getData("Device/SubDeviceList/LShoulderRoll/Position/Sensor/Value")
    LElbowRollTestData = memory.getData("Device/SubDeviceList/LElbowRoll/Position/Sensor/Value")
    LElbowYawTestData = memory.getData("Device/SubDeviceList/LElbowYaw/Position/Sensor/Value")
    LWristYawTestData = memory.getData("Device/SubDeviceList/LWristYaw/Position/Sensor/Value")
    
    LSTMNet_output = LSTMClassificationNet.activate([RShoulderPitchTestData, RShoulderRollTestData, RElbowRollTestData, RElbowYawTestData, RWristYawTestData, 
                                               LShoulderPitchTestData, LShoulderRollTestData, LElbowRollTestData, LElbowYawTestData, LWristYawTestData])
    
    rects = plt.bar(range(6), LSTMNet_output, align='center')
    for rect, h in zip(rects, LSTMNet_output):
        rect.set_height(h)
    
    print "Bar FPS:",1/(time.time()-time_start)
    return rects
    
animBar = animation.FuncAnimation(fig1, animateBar, init_func=initBar,interval=1, blit=True)
animLine = animation.FuncAnimation(fig, animateLine, init_func=initLine,interval=1, blit=True)
plt.show()