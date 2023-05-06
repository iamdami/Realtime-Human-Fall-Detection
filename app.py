# 사람이 파란색으로 탐지되는 문제 수정중

from flask import Flask, request, render_template
import cv2
import os
import torch
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression

app = Flask(__name__)

weights_path = 'FD_best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU 사용 여부에 따라 자동으로 선택

# 클래스 목록 파일 경로
class_path = 'FD.yaml'

# 클래스 목록 불러오기
with open(class_path, 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]

# 모델 로드
model = attempt_load(weights_path, device=device)
model.names = class_names  # 모델에 클래스 이름 설정
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        # 업로드된 비디오 파일 가져오기
        video = request.files['video']
        video_name = video.filename
        video.save(video_name)

        # 비디오 파일 열기
        cap = cv2.VideoCapture(video_name)

        # 비디오 정보 가져오기
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 비디오 저장하기 위한 writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

        while cap.isOpened():
            # 비디오 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break

            # 이미지 색상공간 변경 (BGR -> RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_resized = cv2.resize(frame, (640, 640))
            frame_resized = np.transpose(frame_resized, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            frame_resized = (frame_resized / 255.0).astype(np.float32)  # 이미지를 0~1 범위로 정규화

            results = model(torch.from_numpy(frame_resized).unsqueeze(0).to(device))

            # YOLOv5 모델을 사용하여 쓰러진 사람 탐지
            # 쓰러진 사람이 있는 경우
            if 'Fall Detected' in class_names:
                person_results = results.pandas().xyxy[0] # 모든 클래스에 대한 결과에서 첫 번째 클래스의 탐지 결과만 가져옴
                person_results = person_results[person_results['name'] == 'Fall Detected'] # 'Fall Detected' 클래스에 대한 결과만 추출
                person_results = non_max_suppression(person_results, 0.3, 0.5) # Non-Maximum Suppression 적용

                # 쓰러진 사람이 있는 경우
                if person_results[0] is not None:
                    # 결과에서 바운딩 박스와 확률 추출
                    boxes = person_results[0][:, :4]
                    scores = person_results[0][:, 4]

                    # 바운딩 박스와 확률을 반영하여 원본 이미지에 그리기
                    for box, score in zip(boxes, scores):
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(frame, f'{class_names[0]}: {score:.2f}', (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # 프레임 저장
            out.write(frame)

        # 작업 완료 후 파일 닫기
        cap.release()
        out.release()

        # 생성된 비디오 파일 반환
        return open('output.mp4', 'rb').read()

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
