#!/usr/bin/env bash
# Face models (OpenCV model zoo): YuNet face detector + SFace face recognizer. ~40 MB, CPU only.
# Run once on each machine that runs face_detect (the robot, the VM):  bash models/get_models.sh
set -e
cd "$(dirname "$0")"
BASE=https://github.com/opencv/opencv_zoo/raw/main/models
get() { [ -s "$1" ] && [ "$(stat -c%s "$1")" -gt 100000 ] && { echo "have $1"; return; }
        echo "downloading $1"; curl -fL --retry 3 -o "$1.part" "$BASE/$2" && mv "$1.part" "$1"; }
get face_detection_yunet_2023mar.onnx face_detection_yunet/face_detection_yunet_2023mar.onnx   # needs OpenCV >= 4.8
get face_recognition_sface_2021dec.onnx face_recognition_sface/face_recognition_sface_2021dec.onnx
python3 -c "import cv2; print('OpenCV', cv2.__version__)"
ls -la *.onnx
