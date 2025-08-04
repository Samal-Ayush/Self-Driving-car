import tensorflow.compat.v1 as tf
import cv2
import numpy as np
import colorsys
from ultralytics import YOLO
from typing import List, Tuple
import concurrent.futures
import time

tf.disable_v2_behavior()

class SteeringAnglePredictor:
    def _init_(self, model_path: str):
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, model_path)
        # 'angle_tensor', 'input_tensor', and 'keep_prob' need to be set appropriately with your TF graph

    def predict_angle(self, image) -> float:
        return self.sess.run(self.angle_tensor, feed_dict={
            self.input_tensor: [image],
            self.keep_prob: 1.0
        })[0][0] * 180.0 / 3.14159265

class ImageSegmentation:
    def _init_(self, lane_model_path: str, object_model_path: str):
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
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_lane = executor.submit(self.lane_model.predict, img, conf=0.5)
            future_object = executor.submit(self.object_model.predict, img, conf=0.5)
            lane_results = future_lane.result()
            object_results = future_object.result()
        self._draw_lane_overlay(overlay, lane_results)
        self._draw_object_overlay(overlay, object_results)
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    def _draw_lane_overlay(self, overlay: np.ndarray, lane_results):
        for result in lane_results:
            if result.masks is None:
                continue
            for mask in result.masks.xy:
                points = np.int32([mask])
                cv2.fillPoly(overlay, points, (144, 238, 144))  # Light green

    def _draw_object_overlay(self, overlay: np.ndarray, object_results):
        for result in object_results:
            if result.masks is None:
                continue
            for mask, box in zip(result.masks.xy, result.boxes):
                class_id = int(box.cls[0])
                color = self.colors[class_id]
                points = np.int32([mask])
                cv2.fillPoly(overlay, points, color)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                label = f"{self.object_model.names[class_id]}: {box.conf[0]:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(overlay, (x1, y1 - 20), (x1 + label_w, y1), color, -1)
                cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

class SelfDrivingCarSimulator:
    def _init_(self, steering_model: SteeringAnglePredictor, segmentation_model: ImageSegmentation):
        self.steering_model = steering_model
        self.segmentation_model = segmentation_model
        self.img = None
        self.cols, self.rows = None, None
        self.smoothed_angle = 0.0

    def start_simulation(self, frame_interval: float = 1 / 30):
        self.img = cv2.imread("data/steering_wheel_image.jpg", 0)
        self.cols, self.rows = self.img.shape[1], self.img.shape[0]
        i = 0
        while True:
            start_time = time.time()
            degrees = self.steering_model.predict_angle(self.img)
            segmented_img = self.segmentation_model.process(self.img)
            self.update_display(degrees, segmented_img, self.img)
            i += 1
            if time.time() - start_time < frame_interval:
                time.sleep(frame_interval - (time.time() - start_time))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def update_display(self, degrees, segmented_image, full_image):
        print(f"Predicted steering angle: {degrees:.2f} degrees")
        self.smoothed_angle += 0.2 * pow(abs(degrees - self.smoothed_angle), 2)
        M = cv2.getRotationMatrix2D((self.cols / 2, self.rows / 2), -self.smoothed_angle, 1)
        dst = cv2.warpAffine(full_image, M, (self.cols, self.rows))
        cv2.imshow("Original Frame", full_image)
        cv2.imshow("Segmented Frame", segmented_image)
        cv2.imshow("Steering Wheel", dst)

# Usage example:
if _name_ == "_main_":
    steering_predictor = SteeringAnglePredictor("saved_models/regression_model/model")
    image_segmentation = ImageSegmentation(
        "saved_models/lane_segmentation_model/lane.pt",
        "saved_models/object_detection_model/yolo.pt"
    )
    simulator = SelfDrivingCarSimulator(steering_predictor, image_segmentation)
    simulator.start_simulation()