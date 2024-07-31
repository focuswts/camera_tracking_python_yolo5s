import torch

class YOLOv5Detector:
    def __init__(self, model_name='yolov5s'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.class_names = self.model.names
        self.model.to(self.device).eval()

    def detect(self, frame):
        results = self.model(frame)
        pred = results.pred[0]
        boxes, scores, labels = pred[:, :4], pred[:, 4], pred[:, 5]
        return boxes, scores, labels