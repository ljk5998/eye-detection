# Iris Center Screening Prototype

홍채 중심 좌표를 추적해 안구 운동 스크리닝용 기초 데이터를 수집하는 미니 프로토타입입니다.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

현재 로컬 Python 3.13에서도 실행을 시도할 수 있지만, MediaPipe 호환성 문제가 생기면 Python 3.12 가상환경을 권장합니다.

MediaPipe 최신 Tasks API에서는 `face_landmarker.task` 모델 파일이 필요합니다. 스크립트는 기본적으로 `data/models/face_landmarker.task`가 없으면 자동으로 다운로드를 시도합니다.

## Run

웹캠:

```bash
python src/iris_screening/prototype.py --source 0
```

`--source 0`은 OpenCV의 0번 카메라를 의미합니다. 보통 Mac 내장 웹캠입니다. 외부 웹캠은 `--source 1`, `--source 2`처럼 바꿔 시도할 수 있습니다.

영상 파일:

```bash
python src/iris_screening/prototype.py --source data/raw/sample.mp4
```

CSV만 저장:

```bash
python src/iris_screening/prototype.py --source 0 --no-preview
```

기본 결과 파일은 `data/processed/iris_trace.csv`입니다.
