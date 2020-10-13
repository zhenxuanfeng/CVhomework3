import cv2


def videofacedetect():
    # 创建一个Haar级联分类器 加载.xml分类器文件
    face_cascade = cv2.CascadeClassifier(
        '/opt/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    # 加载视频
    camera = cv2.VideoCapture('00-1.avi')
    fps = camera.get(cv2.CAP_PROP_FPS)
    size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    cv2.namedWindow('Dynamic')
    while True:
        # 读取一帧图像
        ret, frame = camera.read()
        # 判断图片读取成功？
        if ret:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 人脸检测
            faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
            for (x, y, w, h) in faces:
                # 在原图像上人脸周围绘制矩形
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            output.write(frame)
            cv2.imshow('Dynamic', frame)
            # 如果按下q键则退出
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
    camera.release()
    cv2.destroyAllWindows()


videofacedetect()
