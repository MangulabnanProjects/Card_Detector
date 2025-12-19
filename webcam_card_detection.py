import cv2
import sys
import os
from collections import Counter
from ultralytics import YOLO
import pygame  # For audio playback

# Define class names for all playing cards
CLASS_NAMES = [
    "10c", "10d", "10h", "10s",
    "2c", "2d", "2h", "2s",
    "3c", "3d", "3h", "3s",
    "4c", "4d", "4h", "4s",
    "5c", "5d", "5h", "5s",
    "6c", "6d", "6h", "6s",
    "7c", "7d", "7h", "7s",
    "8c", "8d", "8h", "8s",
    "9c", "9d", "9h", "9s",
    "Ac", "Ad", "Ah", "As",
    "Jc", "Jd", "Jh", "Js",
    "Kc", "Kd", "Kh", "Ks",
    "Qc", "Qd", "Qh", "Qs",
]

# Model paths
MODELS = {
    "1": {"name": "Base YOLOv8m", "path": "final_models/yolov8m.pt"},
    "2": {"name": "YOLOv8m Synthetic", "path": "final_models/yolov8m_synthetic.pt"},
    "3": {"name": "YOLOv8m Tuned", "path": "final_models/yolov8m_tuned.pt"},
}


def parse_card_info(card_label):
    """Parse card label to extract value and suit"""
    # Suit mapping - using ASCII text since OpenCV doesn't support Unicode symbols
    suit_map = {
        'c': 'Clubs',
        'd': 'Diamonds',
        'h': 'Hearts',
        's': 'Spades'
    }
    
    # Value is everything except the last character
    value = card_label[:-1].upper()
    suit_code = card_label[-1].lower()
    suit = suit_map.get(suit_code, suit_code)
    
    return value, suit, suit_code


def count_by_suit_and_value(detections):
    """Count cards by suit and by value"""
    suit_counts = Counter()
    value_counts = Counter()
    
    for card in detections:
        value, suit, suit_code = parse_card_info(card)
        suit_counts[suit] += 1
        value_counts[value] += 1
    
    return suit_counts, value_counts


def draw_detections(frame, results, class_names):
    """Draw bounding boxes and labels on frame"""
    detections = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Get class name
            card_label = class_names[cls]
            detections.append(card_label)
            
            # Parse card info
            value, suit, _ = parse_card_info(card_label)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence - show suit symbol based on suit code
            suit_symbols = {'c': 'C', 'd': 'D', 'h': 'H', 's': 'S'}
            suit_symbol = suit_symbols.get(card_label[-1].lower(), '')
            label = f"{value}{suit_symbol} ({conf:.2f})"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
    
    return frame, detections


def draw_stats_panel(frame, suit_counts, value_counts, total_cards, model_name):
    """Draw statistics panel on the frame"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay on the left side
    overlay = frame.copy()
    panel_width = 280
    cv2.rectangle(overlay, (0, 0), (panel_width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    y_offset = 30
    cv2.putText(frame, "CARD DETECTION", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_offset += 30
    
    # Model name
    cv2.putText(frame, f"Model: {model_name}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y_offset += 30
    
    # Total cards
    cv2.putText(frame, f"Total Cards: {total_cards}", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset += 40
    
    # Suit counts
    cv2.putText(frame, "SUITS:", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_offset += 25
    
    suit_colors = {
        'Clubs': (100, 255, 100),      # Green
        'Diamonds': (100, 100, 255),   # Red
        'Hearts': (100, 100, 255),     # Red
        'Spades': (100, 255, 100)      # Green
    }
    
    # Always show all suits (even with 0 count)
    for suit in ['Clubs', 'Diamonds', 'Hearts', 'Spades']:
        count = suit_counts.get(suit, 0)
        color = suit_colors.get(suit, (255, 255, 255))
        # Make the text brighter to ensure visibility
        cv2.putText(frame, f"  {suit}: {count}", (15, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_offset += 25
    
    y_offset += 20
    
    # Value counts
    cv2.putText(frame, "VALUES:", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_offset += 25
    
    # Sort values: A, 2-10, J, Q, K
    value_order = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    for value in value_order:
        count = value_counts.get(value, 0)
        if count > 0:  # Only show values that are detected
            cv2.putText(frame, f"  {value}: {count}", (15, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
            y_offset += 25
    
    # Instructions at the bottom
    y_offset = height - 60
    cv2.putText(frame, "Press 'q' to quit", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 25
    cv2.putText(frame, "Press 'm' to change model", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def select_model():
    """Prompt user to select a model"""
    print("\n" + "="*50)
    print("PLAYING CARD OBJECT DETECTION")
    print("="*50)
    print("\nAvailable Models:")
    for key, model in MODELS.items():
        print(f"{key}. {model['name']}")
    
    while True:
        choice = input("\nSelect a model (1-3): ").strip()
        if choice in MODELS:
            return choice
        print("Invalid choice. Please select 1, 2, or 3.")


def run_webcam_detection(model_path, model_name):
    """Run real-time object detection on webcam feed"""
    print(f"\nLoading model: {model_name}...")
    try:
        model = YOLO(model_path, verbose=False)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Initialize pygame for audio playback
    pygame.mixer.init()
    audio_file = "audio.mp3"
    audio_loaded = False
    
    # Check if audio file exists
    if os.path.exists(audio_file):
        try:
            pygame.mixer.music.load(audio_file)
            audio_loaded = True
            print(f"Audio file '{audio_file}' loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load audio file: {e}")
    else:
        print(f"Warning: Audio file '{audio_file}' not found. Audio will not play.")
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return False
    
    # Set camera resolution for better performance - lower resolution = faster
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPEG codec for speed
    
    print("\n" + "="*50)
    print("Webcam started successfully!")
    print("="*50)
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'm' to change model")
    print("="*50 + "\n")
    
    # Performance optimization variables
    frame_skip = 4  # Process every 4th frame for maximum smoothness
    frame_count = 0
    last_results = None
    last_detections = []
    last_suit_counts = Counter()
    last_value_counts = Counter()
    last_total = 0
    
    # FPS calculation
    import time
    fps_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    # Audio cooldown to prevent spam
    last_audio_time = 0
    audio_cooldown = 2.0  # seconds between audio plays
    last_4s_detected_time = 0  # Track when 4S was last seen
    audio_timeout = 1.0  # Stop audio after 1 second of not seeing 4S
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            frame_count += 1
            
            # Only run inference on every Nth frame for performance
            if frame_count % frame_skip == 0:
                # Run inference with maximum optimization (smaller resolution = much faster)
                results = model(frame, conf=0.5, verbose=False, imgsz=320, half=False)
                
                # Draw detections and get card list
                frame, detections = draw_detections(frame, results, CLASS_NAMES)
                
                # Count by suit and value
                suit_counts, value_counts = count_by_suit_and_value(detections)
                total_cards = len(detections)
                
                # Cache results for skipped frames
                last_results = results
                last_detections = detections
                last_suit_counts = suit_counts
                last_value_counts = value_counts
                last_total = total_cards
                
                # Check if 4 of Spades (4s) is detected and play audio
                if audio_loaded and "4s" in detections:
                    current_time = time.time()
                    last_4s_detected_time = current_time  # Update last seen time
                    
                    if current_time - last_audio_time > audio_cooldown:
                        if not pygame.mixer.music.get_busy():
                            pygame.mixer.music.play()
                            print("🎵 4 of Spades detected! Playing audio...")
                            last_audio_time = current_time
                elif audio_loaded:
                    # 4S not detected - check if we should stop audio
                    current_time = time.time()
                    if pygame.mixer.music.get_busy() and (current_time - last_4s_detected_time > audio_timeout):
                        pygame.mixer.music.stop()
                        print("⏹️ 4 of Spades not detected - Audio stopped.")
            else:
                # Reuse previous detections for skipped frames
                if last_results is not None:
                    frame, _ = draw_detections(frame, last_results, CLASS_NAMES)
                suit_counts = last_suit_counts
                value_counts = last_value_counts
                total_cards = last_total
            
            # Draw statistics panel (lightweight operation)
            frame = draw_stats_panel(frame, suit_counts, value_counts, total_cards, model_name)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter >= 10:
                current_fps = fps_counter / (time.time() - fps_time)
                fps_time = time.time()
                fps_counter = 0
            
            # Display FPS on frame
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (15, frame.shape[0] - 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Playing Card Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('m'):
                print("\nChanging model...")
                cap.release()
                cv2.destroyAllWindows()
                return True  # Signal to change model
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError during detection: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    return False  # Don't change model, just exit


def main():
    """Main function"""
    print("\nWelcome to Playing Card Detection System!")
    print("This system uses YOLOv8 for real-time card detection.\n")
    
    while True:
        # Select model
        choice = select_model()
        model_info = MODELS[choice]
        
        # Run detection
        change_model = run_webcam_detection(model_info["path"], model_info["name"])
        
        if not change_model:
            break
    
    print("\nThank you for using the Card Detection System!")
    print("Goodbye!\n")


if __name__ == "__main__":
    main()
