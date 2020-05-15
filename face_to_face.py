# 摄像头实时人脸识别



import dlib          # 人脸处理的库 Dlib
import numpy as np   # 数据处理的库 numpy
import cv2           # 图像处理的库 OpenCv
import pandas as pd  # 数据处理的库 Pandas
import os
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

# 人脸识别模型，提取128D的特征矢量
facerec = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


# 计算两个128D向量间的欧式距离
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist


# 1. Check 存放所有人脸特征的 csv
if os.path.exists("data/features.csv"):
    path_features_known_csv = "data/features.csv"
    csv_rd = pd.read_csv(path_features_known_csv, header=None)

    # 用来存放所有录入人脸特征的数组
    features_known_arr = []

    # 2. 读取已知人脸数据
    for i in range(csv_rd.shape[0]):
        features_someone_arr = []
        for j in range(0, len(csv_rd.iloc[i])):
            features_someone_arr.append(csv_rd.iloc[i][j])
        features_known_arr.append(features_someone_arr)
    print("Faces in Database：", len(features_known_arr))

    # Dlib 检测器和预测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

    # 创建 cv2 摄像头对象
    cap = cv2.VideoCapture(0)

    # 3. 打开摄像头
    while cap.isOpened():

        flag, img_rd = cap.read()
        faces = detector(img_rd, 0)

        # 待会要写的字体
        font = cv2.FONT_ITALIC

        # 存储当前摄像头中捕获到的所有人脸的坐标/名字
        pos_namelist = []
        name_namelist = []

        kk = cv2.waitKey(1)
        # 按下 q 键退出
        if kk == ord('q'):
            break
        else:
            # 检测到人脸
            if len(faces) != 0:
                # 4. 获取当前捕获到的图像的所有人脸的特征，存储到 features_cap_arr
                features_cap_arr = []
                for i in range(len(faces)):
                    shape = predictor(img_rd, faces[i])
                    # facerec.compute_face_descriptor 特征描述
                    features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))

                # 5. 遍历捕获到的图像中所有的人脸
                for k in range(len(faces)):
                    print("##### camera person", k+1, "#####")
                    # 让人名跟随在矩形框的下方并确定人名的位置坐标
                    # 先默认所有人不认识，是 unknown
                    name_namelist.append("unknown")
                    # 每个捕获人脸的名字坐标
                    pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() +
                                            (faces[k].bottom() - faces[k].top())/4)]))
                    # 对于某张人脸，遍历所有存储的人脸特征
                    e_distance_list = []
                    for i in range(len(features_known_arr)):
                        # 如果 person_X 数据不为空
                        if str(features_known_arr[i][0]) != '0.0':
                            print("with person", str(i + 1), "the e distance: ", end='')
                            e_distance_tmp = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
                            print(e_distance_tmp)
                            e_distance_list.append(e_distance_tmp)
                        else:
                            # 空数据 person_X
                            e_distance_list.append(999999999)
                    similar_person_num = e_distance_list.index(min(e_distance_list))
                    print("Minimum e distance with person", int(similar_person_num)+1)

                    if min(e_distance_list) < 0.4:
                        ####### 在这里修改 person_1, person_2 ... 的名字 ########
                        # 可以在这里改称 david
                        name_namelist[k] = "Person "+str(int(similar_person_num)+1)
                        print("May be person "+str(int(similar_person_num)+1))
                        name_namelist[k] = str("Person " + str(int(similar_person_num) + 1)) \
                            .replace("Person 1", "david") \
                            .replace("Person 2", "pujing") \
                            .replace("Person 3", "lihui")
                    else:
                        print("Unknown person")
                    # 矩形框
                    for kk, d in enumerate(faces):
                        # 绘制矩形框
                        cv2.rectangle(img_rd, tuple([d.left(), d.top()]),
                                      tuple([d.right(), d.bottom()]), (0, 255, 255), 2)
                    print('\n')
                # 6. 在人脸框下面写人脸名字
                for i in range(len(faces)):
                    # img =  cv2ImgAddText(img_rd, name_namelist[i],(pos_namelist[i][0]+60,pos_namelist[i][1]-40), (0, 255, 255) )
                    cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
        print("Faces in camera now:", name_namelist, "\n")
        img = cv2ImgAddText(img_rd, "按键 'q':退出", (20, 400))
        img = cv2ImgAddText(img, "人脸检测", (20, 40),(0, 0, 0))
        img = cv2ImgAddText(img, "人脸数目: " + str(len(faces)), (20, 100), (0, 0, 255))
        cv2.imshow("camera", img)

    cap.release()
    cv2.destroyAllWindows()

else:
    print('##### Warning #####', '\n')
    print("'features_all.py' not found!")
    print("Please run 'get_face.py' and 'save_csv.py' before 'face_to_face.py'", '\n')
    print('##### Warning #####')