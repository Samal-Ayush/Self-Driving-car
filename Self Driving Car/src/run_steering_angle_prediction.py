import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np

class SteeringAnglePredictor:
    def __init__(self, model_path):
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, model_path)
        graph = tf.get_default_graph()
        self.input_tensor = graph.get_tensor_by_name('x:0')
        self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
        self.angle_tensor = graph.get_tensor_by_name('y_out:0')

    def predict_angle(self, image):
        image = image.astype(np.float32)
        image = image / 255.0
        out = self.sess.run(self.angle_tensor, feed_dict={
            self.input_tensor: [image],
            self.keep_prob: 1.0
        })
        # Convert radian output to degrees, assuming your model outputs radians
        return float(out[0][0]) * 180.0 / np.pi

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_steering_angle_pred.py path_to_model path_to_image")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not open image {image_path}")
        sys.exit(1)
    img_cropped = image[-150:, :, :]
    img_resized = cv2.resize(img_cropped, (200, 66))

    predictor = SteeringAnglePredictor(model_path)
    angle = predictor.predict_angle(img_resized)
    print(f"Predicted steering angle: {angle:.2f} degrees")
