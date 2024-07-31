import time
from datetime import datetime
import cv2

class DetectionHandler:
    def __init__(self, detector, notifier, confidence_threshold, detection_interval):
        self.detector = detector
        self.notifier = notifier
        self.confidence_threshold = confidence_threshold
        self.detection_interval = detection_interval
        self.last_detection_time = time.time() - detection_interval
        self.person_detected = False

    def handle_detection(self, frame):
        boxes, scores, labels = self.detector.detect(frame)
        target_class_id = self._get_target_class_id('person')

        if target_class_id is None:
            print(f'Classe "person" não encontrada.')
            return

        person_detected_current_frame = False
        for box, score, label in zip(boxes, scores, labels):
            if int(label) == target_class_id and score > self.confidence_threshold:
                self._draw_box(frame, box, score, label)
                print(f'Detectado {self.detector.class_names[int(label)]} com confiança {score:.2f}')
                person_detected_current_frame = True

        if person_detected_current_frame:
            current_time = time.time()
            if not self.person_detected and (current_time - self.last_detection_time) >= self.detection_interval:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.notifier.send(frame, timestamp)
                self.last_detection_time = current_time
                self.person_detected = True
        else:
            self.person_detected = False

    def _get_target_class_id(self, target_class):
        return next((class_id for class_id, name in self.detector.class_names.items() if name == target_class), None)

    def _draw_box(self, frame, box, score, label):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{self.detector.class_names[int(label)]} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
