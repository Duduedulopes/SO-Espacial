import cv2

backends = {
    "DSHOW": cv2.CAP_DSHOW,
    "MSMF": cv2.CAP_MSMF,
    "ANY": cv2.CAP_ANY,
}

for nome, backend in backends.items():
    for indice in range(4):
        cam = cv2.VideoCapture(indice, backend)
        ok, frame = cam.read()
        if ok:
            h, w = frame.shape[:2]
            print(f"OK  {nome} indice={indice}  {w}x{h}")
        cam.release()

print("varredura terminada")