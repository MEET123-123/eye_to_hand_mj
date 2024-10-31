import time
import numpy as np
import rtde_control
import rtde_receive
from sksurgerynditracker.nditracker import NDITracker
from scipy.spatial.transform import Rotation as R

robot_points_path = "E:/eye_to_hand/robot_positions.txt"
# 读取文件并去除方括号
with open(robot_points_path, 'r') as f:
    data_strings = f.read().replace('[', '').replace(']', '')

# 使用np.fromstring将处理后的字符串转换为数组
# 16*6
data_matrix = np.fromstring(data_strings, sep=' ', dtype=float).reshape(-1, 6)

rtde_r = rtde_receive.RTDEReceiveInterface("192.168.56.1")
rtde_c = rtde_control.RTDEControlInterface("192.168.56.1")

actual_tcp = rtde_r.getTargetTCPPose()
speed_tcp = rtde_r.getTargetTCPSpeed()

print('actual_tcp=',actual_tcp)
print('speed_tcp=',speed_tcp)

def vector2T(pose_vector):
    # vector[0],[1],[2] : x,y,z
    # vector [3][4][5]:rx,ry,rz
    position = pose_vector[:3]  # [x, y, z]
    euler_angles = pose_vector[3:]  # [rx, ry, rz]
    # 创建一个旋转对象，假设欧拉角顺序为'xyz'
    rotation = R.from_euler('xyz', euler_angles, degrees=False)
    # 获取旋转矩阵
    rotation_matrix = rotation.as_matrix()
    # 创建齐次矩阵
    homogeneous_matrix = np.eye(4)  # 创建4x4单位矩阵
    homogeneous_matrix[:3, :3] = rotation_matrix  # 设置旋转部分
    homogeneous_matrix[:3, 3] = position  # 设置平移部分
    return homogeneous_matrix


def T2trans(T):
    return T[:3, 3]


SETTINGS = {
    "tracker type": "polaris",
    #         "romfiles": ["./NDI_Rom/8700339qjx.rom"]
    "romfiles": ["E:/eye_to_hand/tip.rom"]
}
TRACKER = NDITracker(SETTINGS)
TRACKER.start_tracking()

data = []
ur_list = []
ndi_list = []

for i in range(data_matrix.shape[0]):
    move_vector = data_matrix[i,:]
    rtde_c.moveL([move_vector[0], move_vector[1], move_vector[2], move_vector[3],
                  move_vector[4], move_vector[5]], 0.05, 0.05)

    robot_T = vector2T(move_vector)
    robot_trans = T2trans(robot_T)
    raw_track_index = 0
    while (1):
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        ret = tracking[0]
        # 对错误次数进行累加，如果累积超过15次就返回
        # 返回值
        # NDIret = -1
        # NDI_position = np.zeros((4,4))
        if np.isnan(ret[0][0]) == True:
            raw_track_index += 1
        if raw_track_index == 20 and np.isnan(ret[0][0]) == True:
            NDIret = -1
            NDI_position = np.zeros((4, 4))
            break

        NDIret = np.zeros((4, 4))
        if np.isnan(ret[0][0]) == False:
            # 连续取10个值，最终取其平均值
            for j in range(10):
                port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
                NDIret += tracking[0]
            NDIret = NDIret / 10  # 获取的值为4*4齐次矩阵
            NDI_position_trans = T2trans(NDIret)  # 获取平移向量
            break
        if np.isnan(NDI_position_trans[0][0]) == False and NDI_position_trans[0][0] != 0:
            # 机械臂坐标系下的齐次矩阵，NDI坐标系下的齐次矩阵
            tmpdata = [robot_T, NDIret]
            data.append(tmpdata)

            robot_trans = np.transpose(robot_trans)
            NDI_position_trans = np.transpose(NDI_position_trans)

            if len(ur_list) == 0:
                ur_list = robot_trans
                ndi_list = NDI_position_trans
            else:
                ur_list = np.concatenate((ur_list, robot_trans), axis=0)  # n*4矩阵
                ndi_list = np.concatenate((ndi_list, NDI_position_trans), axis=0)  # n*4矩阵
            # i+=1

    TRACKER.stop_tracking()
    TRACKER.close()

rtde_c.disconnect()
rtde_r.disconnect()

'''
# 读取机械臂的位置，并将机械臂移动到指定的位置
for line in f:
    time.sleep(1)
    # read robot TCP positon
    line = line.strip()
    # TODO : move to this position
    time.sleep(2)
    while (1):
        actual_tcp = rtde_r.getTargetTCPPose()
        speed_tcp = rtde_r.getSpeedTcpPose()
        TCP_Pos, robot_homogenous, robot_homogenous_trans, Joint_vel = read_now()
        # 机械臂向量(X,Y,Z,Rx,Ry,Rz)
        # 齐次矩阵
        # 齐次矩阵位移值
        # 关节速度
        if abs(Joint_vel) < 0.5:  # 表明已经运动到指定位置，停止运动
            robot_T = robot_homogenous
            robot_trans = robot_homogenous_trans
            break
        else:
            rtde_c.moveL([TCP_Pos[0], TCP_Pos[1], TCP_Pos[2], TCP_Pos[3],
                          TCP_Pos[4], TCP_Pos[5]], 0.2, 0.2)

    raw_track_index = 0
    # 对错误数值进行计数，如果超过15个错误值就退出NDI跟踪，表明跟踪错误
    while (1):
        port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
        ret = tracking[0]

        # 对错误次数进行累加，如果累积超过15次就返回
        # 返回值
        # NDIret = -1
        # NDI_position = np.zeros((4,4))
        if np.isnan(ret[0][0]) == True:
            raw_track_index += 1
        if raw_track_index == 15 and np.isnan(ret[0][0]) == True:
            NDIret = -1
            NDI_position = np.zeros((4, 4))
            break

        NDIret = np.zeros((4, 4))
        if np.isnan(ret[0][0]) == False:
            # 连续取10个值，最终取其平均值
            for j in range(10):
                port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
                NDIret += tracking[0]
            NDIret = NDIret / 10  # 获取的值为4*4齐次矩阵
            NDI_position_trans = self.get_tip_points(NDIret)  # 获取平移向量
            break

    if np.isnan(NDI_position_trans[0][0]) == False and NDI_position_trans[0][0] != 0:
        # 机械臂坐标系下的齐次矩阵，NDI坐标系下的齐次矩阵
        tmpdata = [robot_T, NDIret]
        data.append(tmpdata)

        robot_trans = np.transpose(robot_trans)
        NDI_position_trans = np.transpose(NDI_position_trans)

        if len(ur_list) == 0:
            ur_list = robot_trans
            ndi_list = NDI_position_trans
        else:
            ur_list = np.concatenate((ur_list, robot_trans), axis=0)  # n*4矩阵
            ndi_list = np.concatenate((ndi_list, NDI_position_trans), axis=0)  # n*4矩阵
        # i+=1
'''