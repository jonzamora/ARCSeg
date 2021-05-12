import cv2

vidcap = cv2.VideoCapture('test_dataset/tool_video_15.mp4')

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        print("CURRENT FRAME:", count)
        cv2.imwrite("test_dataset/test/15_frame"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames

sec = 0
frameRate = 1/25  # 25 frames per second
count=1
success = getFrame(sec)

while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)