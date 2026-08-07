import time
import cv2


def fourcc_legivel(v):
    v = int(v)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_FPS, 30)

# ---- diagnostico ----
print("fourcc pedido :", "MJPG")
print("fourcc real   :", fourcc_legivel(cam.get(cv2.CAP_PROP_FOURCC)))
print("largura       :", cam.get(cv2.CAP_PROP_FRAME_WIDTH))
print("altura        :", cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("exposicao auto:", cam.get(cv2.CAP_PROP_AUTO_EXPOSURE))
print("exposicao     :", cam.get(cv2.CAP_PROP_EXPOSURE))
print()

for _ in range(10):          # descarta os primeiros, a camera se ajusta
    cam.read()

t0 = time.monotonic()
n = 200
for _ in range(n):
    cam.read()
dt = time.monotonic() - t0

print(f"{n} quadros em {dt:.2f}s  ->  {n/dt:.1f} fps")
cam.release()