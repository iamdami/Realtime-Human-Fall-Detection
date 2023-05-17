import cv2

def convert_avi_to_mp4(avi_file, mp4_file):
    # AVI 파일 열기
    cap = cv2.VideoCapture(avi_file)

    # 비디오 정보 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # MP4 파일 저장하기 위한 writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_file, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 저장
        out.write(frame)

    # 파일 닫기
    cap.release()
    out.release()

# AVI 파일을 MP4 파일로 변환
convert_avi_to_mp4('output.avi', 'output.mp4')
