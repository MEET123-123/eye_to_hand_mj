import math
import numpy as np
import rtde_receive
import rtde_control
import cv2

# 使针尖位置不动，使机械臂绕着不同的位置旋转

def EulerAngles2RotationMatrix(theta, format='degree'):
    # 将欧拉角转化为旋转矩阵
    """
    Calculates Rotation Matrix given euler angles.
    :param theta : 1-by-3 list [rx, ry, rz] angle in degree
    1*3的欧拉角矩阵
    :return : RPY角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
    """

    if format == 'degree':
        theta = [i * math.pi / 180.0 for i in theta]

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

# 变成4乘4齐次矩阵
def get_transfer_ret(Rce, Tce):
    ret = np.zeros((4, 4))
    for i in range(3):
        for j in range(3):
            ret[i][j] = Rce[i][j]
        ret[i][3] = Tce[i]
    ret[3][3] = 1
    return ret


def get_rx_tx(homogenous_T):
    # T->RX,TX
    # 通过齐次矩阵获取Rx和Tx
    # Rx为T的3*3矩阵
    # Tx为T的3*1矩阵
    rx = homogenous_T[:3,:3]
    tx = homogenous_T[:3,3]
    return rx, tx

# 初始化
# =rtde_receive.RTDEReceiveInterface("192.168.56.1")
# rtde_c=rtde_control.RTDEControlInterface("192.168.56.1")

def getNewMovPosition(trans_pre_vector,rot_pre_vector,rot_temp_vector):
    # trans_pre_vector : 上一轮的平移向量
    # rot_pre_vector : 上一轮的旋转向量
    # rot_pre_matrix : 上一轮的旋转矩阵
    # tcp_T_pre : 上一轮位置的TCP齐次矩阵

    # rot_temp_vector ： 姿态变换旋转向量
    # rot_temp_matrix : 姿态变换旋转矩阵
    # rot_temp_T : 姿态变换齐次矩阵

    # tcp_T : 当前位置的TCP齐次矩阵
    # tcp_r_matrix : 当前位置的旋转矩阵
    # tcp_r_vector : 当前位置的旋转向量
    # tcp_t_vector : 当前位置的平移向量

    rot_pre_matrix = cv2.Rodrigues(rot_pre_vector)[0]
    tcp_T_pre = get_transfer_ret(rot_pre_matrix, trans_pre_vector)

    # 计算旋转后姿态
    rot_temp_matrix = EulerAngles2RotationMatrix(rot_temp_vector)
    zero = np.array([0, 0, 0]).reshape(3, 1)
    rot_temp_T = get_transfer_ret(rot_temp_matrix, zero)

    tcp_T = np.dot(rot_temp_T, tcp_T_pre)
    tcp_r_matrix, tcp_t_vector = get_rx_tx(tcp_T)
    tcp_r_vector = cv2.Rodrigues(tcp_r_matrix)[0]
    print('tcp_T=',tcp_T)

    # 动作
    # rtde_c.moveL([trans_pre_vector[0], trans_pre_vector[1], trans_pre_vector[2], tcp_r_vector[0], tcp_r_vector[1], tcp_r_vector[2]], 0.2, 0.2)

    # 返回
    # tcp_t_vector: tcp平移向量
    # tcp_r_vector: tcp旋转向量
    return tcp_t_vector,tcp_r_vector


# actual_tcp=rtde_r.getTargetTCPPose()
# actual_q=rtde_r.getActualQ()
# tcp_r_vector = (actual_tcp[3], actual_tcp[4], actual_tcp[5])
# tcp_t_vector = np.array([actual_tcp[0], actual_tcp[1], actual_tcp[2]]).reshape(3, 1)
tcp_r_vector = np.array([0.56,0.22,0.21])
tcp_t_vector = np.array([0.1,0.4,0.9]).reshape((3,1))

np.random.seed(0)
pos_random = np.random.randn(20, 3) * 10

for i in range(20):
    # 生成一系列旋转向量
    rot_temp = pos_random[i,:]
    tcp_t_vector,tcp_r_vector = getNewMovPosition(tcp_t_vector,tcp_r_vector,rot_temp)
