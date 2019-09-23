import os
import cv2

# Windows:
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# Linux:
# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('qlearning.avi', fourcc, 10.0, (1200, 900))

qtable_charts = os.listdir("qtable_charts")
for img_path in qtable_charts:
    frame = cv2.imread(f"qtable_charts/{img_path}")
    out.write(frame)

out.release()