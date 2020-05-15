# 进行人脸录入
# 录入多张人脸


import dlib         # 人脸处理的库 Dlib
import numpy as np  # 数据处理的库 Numpy
import cv2          # 图像处理的库 OpenCv
import os           # 读写文件
import shutil       # 读写文件
from PIL import Image, ImageDraw, ImageFont
# coding=utf-8
# 中文乱码处
def cv2ImgAddText(img, text, mod, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/data_dlib/simsun.ttc", textSize, encoding="utf-8")
    draw.text(mod, text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# OpenCv 调用摄像头
cap = cv2.VideoCapture(0)

# 人脸截图的计数器
cnt_ss = 0

# 存储人脸的文件夹
face_dir = ""

# 保存 faces images 的路径
face_path = "data/data_faces/"


# 1. 新建保存人脸图像文件和数据CSV文件夹

def pre_work_mkdir():
    # 新建文件夹
    if os.path.isdir(face_path):
        pass
    else:
        os.mkdir(face_path)
pre_work_mkdir()


# 2. 删除之前存的人脸数据文件夹
def pre_work_del_old_face_folders():
    # 删除之前存的人脸数据文件夹
    folders_rd = os.listdir(face_path)
    for i in range(len(folders_rd)):
        shutil.rmtree(face_path+folders_rd[i])

    if os.path.isfile("data/features.csv"):
        os.remove("data/features.csv")

# 这里在每次程序录入之前, 删掉之前存的人脸数据
# 如果这里打开，每次进行人脸录入的时候都会删掉之前的人脸图像文件夹
# pre_work_del_old_face_folders()



# 3.如果有之前录入的人脸 在之前 person_x 的序号按照 person_x+1 开始录入
if os.listdir("data/data_faces/"):
    # 获取已录入的最后一个人脸序号
    person_list = os.listdir("data/data_faces/")
    person_num_list = []
    for person in person_list:
        person_num_list.append(int(person.split('_')[-1]))
    person_cnt = max(person_num_list)

# 如果第一次存储或者没有之前录入的人脸, 按照 person_1 开始录入
else:
    person_cnt = 0

# 之后用来控制是否保存图像的 flag
save_flag = 1

# 之后用来检查是否先按 'n' 再按 's'
press_n_flag = 0

while cap.isOpened():
    flag, img_rd = cap.read()
    # print(img_rd.shape)


    kk = cv2.waitKey(1)

    # Dlib 正向人脸检测器
    detector = dlib.get_frontal_face_detector()

    # 人脸检测
    faces = detector(img_rd, 0)

    # 待会要写的字体
    font = cv2.FONT_HERSHEY_TRIPLEX

    # 4. 按下 'n' 新建存储人脸的文件夹
    if kk == ord('n'):
        person_cnt += 1
        face_dir = face_path + "person_" + str(person_cnt)
        os.makedirs(face_dir)
        print('\n')
        print("新建的人脸文件夹 / Create folders: ", face_dir)
        cnt_ss = 0              # 将人脸计数器清零
        press_n_flag = 1        # 已经按下 'n'

    # 检测到人脸
    if len(faces) != 0:
        # 矩形框
        for k, d in enumerate(faces):
            # 计算矩形大小
            # (x,y), (宽度width, 高度height)
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])
            # 计算矩形框大小
            height = (d.bottom() - d.top())
            width = (d.right() - d.left())
            hh = int(height/2)
            ww = int(width/2)

            # 设置颜色
            color_rectangle = (255, 255, 255)

            # 判断人脸矩形框是否超出 480x640
            if (d.right()+ww) > 640 or (d.bottom()+hh > 480) or (d.left()-ww < 0) or (d.top()-hh < 0):
                cv2.putText(img_rd, "OUT OF RANGE", (20, 300), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                color_rectangle = (0, 0, 255)
                save_flag = 0
                if kk == ord('s'):
                    print("请调整位置 / Please adjust your position")
            else:
                color_rectangle = (255, 255, 255)
                save_flag = 1

            cv2.rectangle(img_rd,
                          tuple([d.left() - ww, d.top() - hh]),
                          tuple([d.right() + ww, d.bottom() + hh]),
                          color_rectangle, 2)

            # 根据人脸大小生成空的图像
            img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

            if save_flag:
                # 5. 按下 's' 保存摄像头中的人脸到本地
                if kk == ord('s'):
                    # 检查有没有先按'n'新建文件夹
                    if press_n_flag:
                        cnt_ss += 1
                        for ii in range(height*2):
                            for jj in range(width*2):
                                img_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
                        cv2.imwrite(face_dir + "/img_face_" + str(cnt_ss) + ".jpg", img_blank)
                        print("写入本地 / Save into：", str(face_dir) + "/img_face_" + str(cnt_ss) + ".jpg")
                    else:
                        print("请在按 'S' 之前先按 'N' 来建文件夹 / Please press 'N' before 'S'")
    # 显示人脸数
    # cv2.putText(img_rd, "Faces: " + str(len(faces)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    img = cv2ImgAddText(img_rd, "人脸数目: " + str(len(faces)), (20, 100))
    # 添加说明
    img = cv2ImgAddText(img, "按键 'Q':退出", (20, 450))
    img = cv2ImgAddText(img, "按键 'N':新建文件夹", (20, 350))
    img = cv2ImgAddText(img, "按键 'S':保存", (20, 400))
    img = cv2ImgAddText(img, "人脸检测", (20, 40))

    # 6. 按下 'q' 键退出
    if kk == ord('q'):
        break

    # 如果需要摄像头窗口大小可调
    # cv2.namedWindow("camera", 0)

    cv2.imshow("camera", img)

# 释放摄像头
cap.release()
cv2.destroyAllWindows()