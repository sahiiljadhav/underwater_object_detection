import cv2
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np

class DetectionSystem:
    def __init__(self):
        # Load pre-trained model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        
        # Define restricted area (center of frame)
        self.restricted_area = {
            'x_min': 0.3,  # 30% from left
            'x_max': 0.7,  # 70% from left
            'y_min': 0.3,  # 30% from top
            'y_max': 0.7   # 70% from top
        }
        
        # Camera parameters (you may need to adjust these)
        self.focal_length = 1000  # Focal length in pixels
        self.known_width = 0.4    # Known width of an average object in meters
        
    def estimate_distance(self, apparent_width):
        """
        Estimate distance using the formula:
        distance = (known_width * focal_length) / apparent_width
        """
        distance = (self.known_width * self.focal_length) / apparent_width
        return distance
        
    def calibrate_camera(self):
        """
        Simple calibration routine
        Hold an object of known width (e.g., a paper that's 0.2 meters wide) 
        at a known distance (e.g., 1 meter)
        """
        print("=== Camera Calibration ===")
        print("1. Hold an object of known width (e.g., A4 paper) at 1 meter distance")
        print("2. Draw a box around it using the mouse")
        print("3. Press 'c' when done, 'q' to skip calibration")
        
        cap = cv2.VideoCapture(0)
        bbox = None
        drawing = False
        start_point = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal bbox, drawing, start_point
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
                
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                frame_copy = frame.copy()
                cv2.rectangle(frame_copy, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow('Calibration', frame_copy)
                
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                bbox = (start_point[0], start_point[1], x, y)
                frame_copy = frame.copy()
                cv2.rectangle(frame_copy, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow('Calibration', frame_copy)
        
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', mouse_callback)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if bbox is None:
                cv2.imshow('Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and bbox is not None:
                # Calculate focal length using known distance (1 meter)
                apparent_width = abs(bbox[2] - bbox[0])
                self.focal_length = (apparent_width * 1.0) / self.known_width
                print(f"Calibration complete! Focal length: {self.focal_length}")
                break
            elif key == ord('q'):
                print("Skipping calibration...")
                break
        
        cap.release()
        cv2.destroyWindow('Calibration')
    
    def detect_objects(self, frame):
        # Convert frame to tensor
        tensor = F.to_tensor(frame).unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(tensor)
        
        # Process results
        boxes = predictions[0]['boxes']
        scores = predictions[0]['scores']
        
        # Filter good detections
        good_detections = []
        height, width = frame.shape[:2]
        
        for box, score in zip(boxes, scores):
            if score > 0.5:  # 50% confidence threshold
                x1, y1, x2, y2 = box.numpy()
                
                # Get center point
                center_x = (x1 + x2) / (2 * width)
                center_y = (y1 + y2) / (2 * height)
                
                # Check if in restricted area
                if (self.restricted_area['x_min'] <= center_x <= self.restricted_area['x_max'] and
                    self.restricted_area['y_min'] <= center_y <= self.restricted_area['y_max']):
                    
                    # Calculate distance
                    apparent_width = abs(x2 - x1)
                    distance = self.estimate_distance(apparent_width)
                    
                    good_detections.append({
                        'box': box.numpy(),
                        'score': score.item(),
                        'distance': distance
                    })
        
        return good_detections
    
    def draw_results(self, frame, detections):
        output = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw restricted area
        x_min = int(self.restricted_area['x_min'] * width)
        x_max = int(self.restricted_area['x_max'] * width)
        y_min = int(self.restricted_area['y_min'] * height)
        y_max = int(self.restricted_area['y_max'] * height)
        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Draw detections
        for det in detections:
            box = det['box']
            cv2.rectangle(output, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (0, 0, 255), 2)
            
            # Add confidence score and distance
            text = f"Dist: {det['distance']:.2f}m Conf: {det['score']:.2f}"
            cv2.putText(output, text, 
                       (int(box[0]), int(box[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return output

# Main execution
if __name__ == "__main__":
    # Initialize
    detector = DetectionSystem()
    
    # Run calibration
    detector.calibrate_camera()
    
    # Start detection
    cap = cv2.VideoCapture(0)
    print("Starting detection... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break
            
        # Detect objects
        detections = detector.detect_objects(frame)
        
        # Draw results
        output_frame = detector.draw_results(frame, detections)
        
        # Show results
        cv2.imshow('Detection System', output_frame)
        
        # Print detections with distances
        if detections:
            print("\nDetected objects:")
            for i, det in enumerate(detections):
                print(f"Object {i+1}: Distance = {det['distance']:.2f}m, "
                      f"Confidence = {det['score']:.2f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()