#! /usr/bin/env python3

from PIL import Image
import cv2
import numpy as np

# GIFファイルの読み込み
gif_path = 'animation.gif'
img = Image.open(gif_path)

# フレームをリストに保存
frames = []
try:
    while True:
        frame = np.array(img.convert('RGB'))  # RGB形式に変換
        frames.append(frame)
        img.seek(len(frames))  # 次のフレームに進む
except EOFError:
    pass

# OpenCVを使って表示
for frame in frames:
    cv2.imshow('GIF', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):  # 'q'キーで終了
        break

cv2.destroyAllWindows()
