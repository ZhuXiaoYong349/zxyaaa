import cv2
import numpy as np
import glob

# 定义拉普拉斯滤波器
laplacian_filter = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]], dtype=np.float32)

# 设置输入和输出路径
input_path = "C:\\Users\\ZhuXi\\Desktop\\UAVdataset\\VOCdevkit\\VOC2007\\JPEGImages/*.jpg"
output_path = "C:\\Users\\ZhuXi\\Desktop\\UAVdataset\\VOCdevkit\\VOC2007\\out/"

# 获取输入路径下的所有图像文件
image_files = glob.glob(input_path)

# 遍历每个图像文件
for file in image_files:
    # 读取图像
    image = cv2.imread(file)
    
    # 图像锐化
    sharpened_image = cv2.filter2D(image, -1, laplacian_filter)
    
    # 保存锐化后的图像
    file_name = file.split("/")[-1]  # 获取文件名
    save_path = output_path  + file_name
    cv2.imwrite(save_path, sharpened_image)
    print('file_name')
print('ok')