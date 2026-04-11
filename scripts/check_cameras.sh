#!/bin/bash
# 拍照檢查攝影機角度
# 用法: bash scripts/check_cameras.sh
# 照片存在 /tmp/cam_video*.jpg

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR/Isaac-GR00T"
source .venv/bin/activate

python3 -c "
import cv2, sys

cameras = {0: 'wrist', 2: 'front'}
for idx, name in cameras.items():
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            path = f'$PROJECT_DIR/camera_check/{name}.jpg'
            cv2.imwrite(path, frame)
            print(f'{name} (video{idx}): OK -> {path}')
        else:
            print(f'{name} (video{idx}): opened but no frame')
        cap.release()
    else:
        print(f'{name} (video{idx}): cannot open')

print()
print('Done. Check images:')
print(f'  eog $PROJECT_DIR/camera_check/front.jpg $PROJECT_DIR/camera_check/wrist.jpg')
"
