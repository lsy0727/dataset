# dataset

### collect_data.py -> .exe파일로 변환
pyinstaller .\collect_data.py --name OAKDDataCollector --onedir --noconfirm --clean --contents-directory _internal --collect-all depthai --collect-all cv2 --collect-all numpy --collect-binaries cv2
