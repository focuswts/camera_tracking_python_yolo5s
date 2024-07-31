import cv2

from yolov5_detector import YOLOv5Detector
from discord_notifier import DiscordNotifier
from detection_handler import DetectionHandler

# Configurações
WEBHOOK_URL = 'DISCORD_WEBHOOK_URL'
CONFIDENCE_THRESHOLD = 0.5
DETECTION_INTERVAL = 30

def main():
    # Inicializar componentes
    detector = YOLOv5Detector()
    notifier = DiscordNotifier(WEBHOOK_URL)
    handler = DetectionHandler(detector, notifier, CONFIDENCE_THRESHOLD, DETECTION_INTERVAL)

    # Abrir o vídeo da webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        handler.handle_detection(frame)

        # Mostrar o frame com as detecções
        cv2.imshow('YOLOv5 Detection', frame)

        # Sair com 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
