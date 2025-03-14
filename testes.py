# import cv2
# import openpyxl
# import pandas as pd


# #print("Todas as bibliotecas foram instaladas com sucesso!")
# import cv2

# # Inicializa a câmera (0 = webcam padrão)
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()  # Captura o frame
#     if not ret:
#         break

#     cv2.imshow("Câmera ao Vivo", frame)  # Mostra o vídeo em tempo real

#     # Pressione 'q' para sair
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()  # Libera a câmera
# cv2.destroyAllWindows()  # Fecha a janela


import cv2
import mediapipe as mp

# Inicializa o detector de mãos
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
maos = mp_maos.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inicializa a câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converte a imagem para RGB (Mediapipe usa RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa a imagem para detectar a mão
    resultado = maos.process(frame_rgb)

    # Se detectar uma mão
    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            # Desenha os pontos da mão na tela
            mp_desenho.draw_landmarks(frame, hand_landmarks, mp_maos.HAND_CONNECTIONS)

    # Exibe o resultado
    cv2.imshow("Detecção de Mão", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
