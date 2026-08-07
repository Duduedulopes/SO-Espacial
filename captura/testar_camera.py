import cv2

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Pedimos. Não é garantia.
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cam.set(cv2.CAP_PROP_FPS, 30)

# O que a câmera realmente aceitou:
print("largura:", cam.get(cv2.CAP_PROP_FRAME_WIDTH))
print("altura :", cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("fps    :", cam.get(cv2.CAP_PROP_FPS))

while True:
    ok, frame = cam.read()
    if not ok:
        print("falha ao ler quadro")
        break

    cv2.imshow("teste - ESC para sair", frame)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()