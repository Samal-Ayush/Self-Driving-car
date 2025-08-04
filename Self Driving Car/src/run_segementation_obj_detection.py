import sys
import cv2
from ultralytics import YOLO
import numpy as np
from run_steering_angle_pred import SteeringAnglePredictor  # If using its utilities elsewhere
import colorsys
from typing import List, Tuple

class ImageSegmentation:
    def __init__(self, lane_model_path: str, object_model_path: str):
        self.lane_model = YOLO(lane_model_path)
        self.object_model = YOLO(object_model_path)
        self.colors = self._generate_colors(len(self.object_model.names))

    @staticmethod
    def _generate_colors(num_classes: int) -> List[Tuple[int, int, int]]:
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors.append(tuple(int(x * 255) for x in rgb))
        return colors

    def process(self, img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        overlay = img.copy()
        lane_results = self.lane_model.predict(img, conf=0.5)
        object_results = self.object_model.predict(img, conf=0.5)
        self._draw_lane_overlay(overlay, lane_results)
        self._draw_object_overlay(overlay, object_results)
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    def _draw_lane_overlay(self, overlay: np.ndarray, lane_results):
        for result in lane_results:
            if getattr(result, "masks", None) is None:
                continue
            for mask in result.masks.xy:
                points = np.int32([mask])
                cv2.fillPoly(overlay, points, (144, 238, 144))  # Light green

    def _draw_object_overlay(self, overlay: np.ndarray, object_results):
        for result in object_results:
            if getattr(result, "masks", None) is None:
                continue
            for mask, box in zip(result.masks.xy, result.boxes):
                class_id = int(box.cls[0])
                color = self.colors[class_id]
                points = np.int32([mask])
                cv2.fillPoly(overlay, points, color)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                label = f"{self.object_model.names[class_id]}: {box.conf[0]:.2f}"
                (label_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(overlay, (x1, y1 - 20), (x1 + label_w, y1), color, -1)
                cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_segmentation_obj.py lane_model_path object_model_path input_image_path")
        sys.exit(1)

    lane_model_path = sys.argv[1]
    object_model_path = sys.argv[2]
    img_path = sys.argv[3]

    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not open image {img_path}")
        sys.exit(1)

    segmenter = ImageSegmentation(lane_model_path, object_model_path)
    overlay_img = segmenter.process(img)

    cv2.imshow("Segmentation Overlay", overlay_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
