import numpy as np
import math
from sksurgerynditracker.nditracker import NDITracker
import socket
import time
import cv2
import pandas as pd
import rtde_control
import rtde_receive

class matrix_transform:

    default_r = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    default_t = np.array([0,0,0])
    default_homogeneous = np.eye(4)

    def __init__(self,r=default_r,t=default_t,homogeneous=default_homogeneous,stream_data = None):
        self.homogeneous = homogeneous
        self.r = r
        self.t = t
        self.stream_data = stream_data
        self.HOST = '192.168.56.1'  # The remote host
        self.PORT = 30004
        # stream_data_fomat(time_serial):data([0]:ur_data,[1]:ndi_data)

    def get_coordinate_transform(self):
        ret=[]
        for i in range(len(self.stream_data)-1):
            data1,data2=self.stream_data[i],self.stream_data[i+1]
            # 按照顺序存储ur，ndi的数据
            ur1,ndi1=data1[0],data1[1]
            ur2,ndi2=data2[0],data2[1]
            A=np.dot(ndi2,np.linalg.inv(ndi1))
            B=np.dot(ur2,np.linalg.inv(ur1))
            # A * ndi1 = ndi2
            # B * ur1 = ur2
            # 获取姿态变换矩阵 A,B
            # V(w wb) = C(w b) * V(b wb)
            # V(w wb):坐标V(wb)在w坐标系下的投影
            # V(b wb):坐标V(wb)在b坐标系下的投影
            # C(w b):坐标系变换
            # 获取变换坐标系
            tmp=[A,B]
            ret.append(tmp)
        return ret

    def get_tip_points(self,T_homo):
        # 可以将ndi的第三列数据(x,y,z)存储为齐次坐标系(x,y,z,1)
        # 并四舍五入到小数点后3位数
        tip_points_trans = T_homo[:,3]
        tip_points_trans = round(tip_points_trans, 3)
        return tip_points_trans

    def r_t_homogenous(self,r,t):
        # RX(3,3),TX(3,1)->T
        # 使用Rx旋转矩阵和Tx平移矩阵构造4*4的齐次矩阵
        # [Rx(0,0) Rx(0,1) Rx(0,2) Tx(0)]
        # [Rx(1,0) Rx(1,1) Rx(1,2) Tx(1)]
        # [Rx(2,0) Rx(2,1) Rx(2,2) Tx(2)]
        # [0       0       0       1    ]
        ret = np.zeros((4, 4))
        for i in range(3):
            for j in range(3):
                ret[i][j] = r[i][j]
            ret[i][3] = t[i][0]
        ret[3][3] = 1
        return ret

    def get_rx_tx(self,homogenous_T):
        # T->RX,TX
        # 通过齐次矩阵获取Rx和Tx
        # Rx为T的3*3矩阵
        # Tx为T的3*1矩阵
        rx = homogenous_T[:3,:3]
        tx = homogenous_T[:3,3]
        return rx, tx

    def transpose(self,matrix):  # 转置
        # 获取矩阵转置
        return np.transpose(matrix)

    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self,R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def EulerAngles2RotationMatrix(self,theta, format='degree'):
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
    
    def RotationMatrixToEulerAngles(self,R):

        # 欧拉角做旋转矩阵时，中间坐标系如果出现旋转90°时
        # 会出现万向节死锁现象，损失一个自由度，导致旋转矩阵奇异
        # Rank = 2
        # 矩阵的行列式值为0

        assert (self.isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            print("y轴90度-死锁状态")
        #         x = math.atan2(-R[1,2], R[1,1])
        #         y = math.atan2(-R[2,0], sy)
        #         z = 0
        angle = np.array([x, y, z]) / math.pi * 180
        return angle

    def enable(self):
        # 机械臂使能
        pass

    def disable(self):
        # 断开机械臂
        pass

    def read_now(self):

        # getTargetTcpPose
        # return [X, Y, Z, Rx, Ry, Rz]

        # TODO
        # data_read = getTargetTcpPose()
        # data_read：n个时刻的总数居(n*6)
        # 每个数据包含6个值
        # [0 : 3]平移量
        # [4 : 6]欧拉角
        data_read = self.getTargetTcpPose()
        temp_joint_vel = self.getTargetJointVel()

        r_euler = data_read[3:]
        r_matrix = self.EulerAngles2RotationMatrix(r_euler)
        t_matrix = data_read[:3]

        homogenous_matrix = self.r_t_homogenous(r_matrix, t_matrix)
        homogenous_t = np.zeros((4, 1))
        homogenous_t[0][0], homogenous_t[1][0], homogenous_t[2][0] = round(t_matrix[0], 3), round(t_matrix[1], 3), round(t_matrix[2], 3)
        homogenous_t[3][0] = 1

        return data_read, homogenous_matrix, homogenous_t, temp_joint_vel
        # 机械臂6自由度数据、齐次矩阵、齐次矩阵位移值、当前速度

    def eye_hand_calib(self,robot_points_path):
        #######################初始化NDI##############################

        # TODO:修改ROM文件
        SETTINGS = {
            "tracker type": "polaris",
            #         "romfiles": ["./NDI_Rom/8700339qjx.rom"]
            "romfiles": ["NDI_Rom/8700340_2022_11_21.rom"]
        }
        TRACKER = NDITracker(SETTINGS)
        TRACKER.start_tracking()

        #######################初始化机械臂##############################
        # 越疆机械臂连接
        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # s.connect((self.HOST, self.PORT))

        # UR机械臂连接
        rtde_c = rtde_control.RTDEControlInterface(self.HOST)

        # TODO:
        f = open(robot_points_path, "r")
        data = []
        ur_list, ndi_list = [], []
        for line in f:
            time.sleep(1)
            # read robot TCP positon
            line = line.strip()
            # TODO : move to this position
            time.sleep(2)
            while (1):
                TCP_Pos, robot_homogenous, robot_homogenous_trans,Joint_vel = self.read_now()
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
                    NDIret = NDIret / 10 # 获取的值为4*4齐次矩阵
                    NDI_position_trans = self.get_tip_points(NDIret) #获取平移向量
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

        ur_list = np.reshape(ur_list, (4,-1)) # 形状变为4*n
        ndi_list = np.reshape(ndi_list, (4,-1))

        ndi_ur = np.dot(ndi_list, np.linalg.pinv(ur_list)) # ndi到ur的转换矩阵
        ndi_ur[3][0], ndi_ur[3][1], ndi_ur[3][2] = 0, 0, 0  # 最后一行直接置为0

        # TODO: get the matrix transform
        rx_vector = np.zeros((3, 1))
        for i in range(len(data)):
            T_tip2B = data[i][0]  # 机械臂下针尖
            T_w2tip = np.linalg.inv(data[i][1])
            T_tip2w = np.dot(ndi_ur, T_tip2B)
            T_tipB2W = np.dot(T_w2tip, T_tip2w)
            rx, tx = self.get_rx_tx(T_tipB2W)
            rx = cv2.Rodrigues(rx)[0]  # 旋转矩阵转为旋转向量，做旋转向量的均值
            rx_vector += rx
        rx_vector = rx_vector / len(data)
        rx_matrix = cv2.Rodrigues(rx_vector)[0]
        # 旋转向量转回旋转矩阵，这一步不涉及机械臂的读取和交互，因此不需要考虑使用欧拉角还是旋转向量
        t_vector = np.zeros((4, 1))
        t_vector[3][0] = 1
        # TODO:
        T_tipB2W = self.get_transfer_ret(rx_matrix, t_vector)
        return T_tipB2W,ndi_ur,ndi_list,ur_list,data

    def getTargetTcpPose(self):
        # TODO:
        # return [X, Y, Z, Rx, Ry, Rz]
        result_test = np.array([0,0,0,0,0,0])
        actual_tcp = rtde_r.getTargetTCPPose()
        return result_test

    def getTargetJointVel(self):
        # TODO:
        # return [Vx, Vy, Vz, Wrx, Wry, Wrz]
        result_test = np.array([0,0,0,0,0,0])
        return result_test

    def process(self,data):
        data = list(data)
        for i in range(len(data)):
            data[i] = round(data[i], 6)
        return data

    def write_data_read(self,file_path):
        # 手眼标定时写入的数据
        # getTargetTcpPose
        # return [X, Y, Z, Rx, Ry, Rz]
        # TODO
        # data_read = getTargetTcpPose()
        # data_read：n个时刻的总数居(n*6)
        # 每个数据包含6个值
        # [0 : 3]平移量
        # [4 : 6]欧拉角
        dataList = []

        # 使机械臂的TCP不动，移动其关节位置，使其变换10次
        for i in range(10):
            data_read, homogenous_matrix, homogenous_t, temp_joint_vel = self.read_now()
            data_read = round(data_read, 6)
            dataList.append(data_read)
            time.sleep(5)

        # 手眼标定时的数据
        with open(file_path, "w") as f:
            for i in range(len(dataList)):
                f.write(str(dataList[i]))
                f.write('\n')

    def calibration(self):
        self.enable() #机械臂使能
        robot_pos_filepath_train = "robot_pos_train.txt"
        robot_pos_filepath_val = "robot_pos_val.txt"
        print("使机械臂TCP位置不动，移动其关节位置使其到达10个位置,得到训练数据")
        self.write_data_read(robot_pos_filepath_train)
        print("使机械臂TCP位置不动，移动其关节位置使其到达10个位置,得到验证数据")
        self.write_data_read(robot_pos_filepath_val)

        T_tipB2W, ndi_ur, ndi_list, ur_list,_ = self.eye_hand_calib(robot_pos_filepath_train)
        # 机械臂基坐标系下针尖到NDI世界坐标系的变换矩阵，NDI坐标系到UR坐标系的变化矩阵，NDI坐标系下观测到的针尖坐标，UR坐标系下观测到的针尖坐标
        # 训练时得到其坐标系转换矩阵

        # 验证误差和精度
        # 验证时得到其ur和NDI的数据，使用转换矩阵对应后，判断其精度
        _,_,_,_,data = self.eye_hand_calib(robot_pos_filepath_val)

        test_data = []
        # ur_data(x,y,z);ndi_data(x,y,z)
        for i in range(len(data)):
            tmpdata = data[i]
            robot_T, NDI_T = tmpdata[0], tmpdata[1]  # 将data的信息解码
            # 4*4, 4*4
            valid_pos_B = robot_T[:,3]
            valid_pos_tip_w = np.dot(ndi_ur, valid_pos_B) # 4*1

            # get x,y,z trans
            valid_pos_tip_w = np.transpose(valid_pos_tip_w)
            valid_pos_tip_w = valid_pos_tip_w[:3]
            valid_pos_tip_ndi = NDI_T[3,:3]
            valid_pos_tip_ndi = np.transpose(valid_pos_tip_ndi)
            temp = [valid_pos_tip_w, valid_pos_tip_ndi]
            test_data.append(temp)


        result = pd.DataFrame(np.array(test_data))
        result.columns = ["x_ur", "y_ur", "z_ur", "x_ndi", "y_ndi", "z_ndi"]
        error = ((result["x_ur"] - result["x_ndi"]) ** 2 + (result["y_ur"] - result["y_ndi"]) ** 2 + (
                    result["z_ur"] - result["z_ndi"]) ** 2) ** 0.5
        result["error"] = error
        print(f"error.mean():{error.mean()}\terror.std():{error.std()}")


#from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
from PIL import Image

def convert_tif_to_jpg(tif_file_path, jpg_file_path, quality=95):
    """
    Convert a TIFF file to a JPEG file.

    :param tif_file_path: Path to the TIFF file.
    :param jpg_file_path: Path to the output JPEG file.
    :param quality: JPEG quality, an integer between 1 (worst) and 95 (best). Default is 85.
    """
    try:
        # 打开TIF文件
        with Image.open(tif_file_path) as img:
            # 转换图像到RGB，因为JPG不支持透明通道
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGB')
            # 保存为JPG文件
            img.save(jpg_file_path, 'JPEG', quality=quality)
            print(f"Converted {tif_file_path} to {jpg_file_path} with quality {quality}.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    #test = matrix_transform()
    #print(test.EulerAngles2RotationMatrix([120, 0, 0]))
    #test.enable()
    HOST = '192.168.56.1'
    PORT = 30004
    # 初始化
    rtde_r = rtde_receive.RTDEReceiveInterface(HOST)
    rtde_c = rtde_control.RTDEControlInterface(HOST)
    # 接收
    actual_tcp = rtde_r.getTargetTCPPose()
    # [(x,y,z,rx,ry,rz)]
    actual_q = rtde_r.getActualQ()
    print('actual_tcp=',actual_tcp)
    print('actual_q=',actual_q)

    #tif_file = 'H:/CVC-ClinicDB/Ground Truth/1.tif'
    #jpg_file = 'H:/CVC-ClinicDB/process/1.jpg'
    #convert_tif_to_jpg(tif_file, jpg_file)

    # The classes here is the total classes in the dataset, for COCO dataset we have 80 classes
    #convert_segment_masks_to_yolo_seg("H:/CVC-ClinicDB/Ground Truth", "H:/CVC-ClinicDB/process", classes=1)
