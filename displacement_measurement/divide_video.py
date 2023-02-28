import cv2

video_path = 'V:/2022SHM-dataset/project2/case3/data_vision.MTS'
image_path = 'V:/2022SHM-dataset/project2/case3_images'

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 创建一个文件夹来存放图片

# 初始化帧计数器
count = 0

# 循环读取视频帧
while cap.isOpened():
    # 读取一帧
    ret, frame = cap.read()
    # 如果没有读到，则退出循环
    if not ret:
        break
    # 如果是第5的倍数，则保存图片
    # 构造图片文件名
    img_name = "{}/frame{}.jpg".format(image_path, count)
    # 保存图片
    cv2.imwrite(img_name, frame)
    print("Saved {}".format(img_name))
    # 更新帧计数器
    count += 1

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
