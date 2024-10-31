import time
import numpy as np
import rtde_control
import rtde_receive

rtde_r = rtde_receive.RTDEReceiveInterface("192.168.56.1")

dataList = []

# 将TCP移动到针尖位置处时，启动该程序，运行后，每隔5s，自己变换一次位置，程序会记录机械臂的TCP位置。

for i in range(20):
    actual_tcp = rtde_r.getTargetTCPPose()
    actual_tcp = np.array(actual_tcp)
    # [x,y,z,rx,ry,rz]
    actual_tcp = np.round(actual_tcp, 6)
    dataList.append(actual_tcp)
    time.sleep(5)

print('datalist=',dataList)
# 手眼标定时的数据
with open("E:/eye_to_hand/robot_positions.txt", "w") as f:
    for i in range(len(dataList)):
        f.write(str(dataList[i]))
        f.write('\n')