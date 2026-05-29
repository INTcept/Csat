"""
Screen Action Recognition System

Adapts the existing camera-based action recognition pipeline to use Windows
screen capture (via mss) as the frame source instead of a physical webcam.

Detects actions ("walking", "pointing", "beckoning") in videos playing on screen
using MediaPipe Holistic keypoint extraction and LSTM classification.
"""

import os
import logging

import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
import mss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Action labels for the screen-based recognition system
ACTIONS = np.array(["walking", "pointing", "beckoning"])

# Data collection parameters
DATA_PATH = "MP_Data"
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30

# Model parameters
MODEL_PATH = "screen_action.h5"
NUM_KEYPOINTS = 1662

# MediaPipe setup
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Compatibility shim: older mediapipe versions exposed FACE_CONNECTIONS on the
# holistic module. Newer versions moved it to face_mesh as FACEMESH_TESSELATION.
if not hasattr(mp_holistic, 'FACE_CONNECTIONS'):
    mp_holistic.FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def safe_extract_keypoints(frame: np.ndarray | None, holistic_model) -> np.ndarray:
    """
    Wrapper that handles null/empty frames gracefully.

    Args:
        frame: BGR numpy array or None.
        holistic_model: MediaPipe Holistic instance.

    Returns:
        1662-element float64 numpy array (zeros if frame is null/empty).
    """
    zero_keypoints = np.zeros(NUM_KEYPOINTS, dtype=np.float64)

    # Handle None input
    if frame is None:
        return zero_keypoints

    # Handle empty arrays
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return zero_keypoints

    try:
        _, results = mediapipe_detection(frame, holistic_model)
        keypoints = extract_keypoints(results)
        return keypoints.astype(np.float64)
    except Exception as e:
        logger.warning(f"Keypoint extraction failed: {e}")
        return zero_keypoints


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


class ScreenCapture:
    """Captures frames from PC display using mss library."""

    def __init__(self, region: dict | None = None):
        """
        Initialize screen capture.

        Args:
            region: Optional dict with keys 'top', 'left', 'width', 'height'
                    (non-negative integers, width/height >= 1).
                    If None, captures entire primary monitor.

        Raises:
            ImportError: If mss is not installed.
            ValueError: If region parameters are invalid (non-integer, negative,
                       zero/negative dimensions, or exceed monitor bounds).
        """
        # Verify mss is available at runtime
        try:
            import mss as _mss
        except ImportError:
            raise ImportError(
                "mss is required for screen capture. Install it with: pip install mss"
            )

        # Create mss instance and get primary monitor info
        self._sct = _mss.mss()
        primary_monitor = self._sct.monitors[1]  # monitors[0] is "all", [1] is primary

        if region is None:
            # Default to full primary monitor
            self._monitor = {
                "top": primary_monitor["top"],
                "left": primary_monitor["left"],
                "width": primary_monitor["width"],
                "height": primary_monitor["height"],
            }
        else:
            # Validate region parameters
            self._validate_region(region, primary_monitor)
            self._monitor = {
                "top": region["top"],
                "left": region["left"],
                "width": region["width"],
                "height": region["height"],
            }

        self._consecutive_failures = 0

    def grab_frame(self) -> np.ndarray:
        """
        Capture a single frame from the configured region.

        Returns:
            numpy array with shape (height, width, 3), dtype uint8, BGR channel order.

        Raises:
            RuntimeError: If 30 consecutive capture attempts fail.
        """
        try:
            screenshot = self._sct.grab(self._monitor)
            # Convert mss screenshot to numpy array (BGRA format)
            frame = np.array(screenshot, dtype=np.uint8)
            # Convert BGRA → BGR by dropping alpha channel and reordering
            # mss returns BGRA, so we take B, G, R channels (indices 0, 1, 2)
            frame = frame[:, :, :3]  # Drop alpha channel (BGRA → BGR)
            # Reset consecutive failure counter on success
            self._consecutive_failures = 0
            return frame
        except Exception as e:
            self._consecutive_failures += 1
            logger.warning(
                f"Frame capture failed (attempt {self._consecutive_failures}/30): {e}"
            )
            if self._consecutive_failures >= 30:
                raise RuntimeError(
                    "Screen capture is unavailable: 30 consecutive capture attempts failed."
                )
            raise

    def release(self) -> None:
        """Release the mss screen capture handle."""
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def _validate_region(self, region: dict, monitor: dict) -> None:
        """
        Validate capture region parameters against monitor bounds.

        Args:
            region: Dict with 'top', 'left', 'width', 'height' keys.
            monitor: Primary monitor info from mss.

        Raises:
            ValueError: If any parameter is invalid.
        """
        required_keys = ("top", "left", "width", "height")
        for key in required_keys:
            if key not in region:
                raise ValueError(
                    f"Region must include '{key}'. "
                    f"Expected keys: top, left, width, height."
                )

        # Validate types - must be integers
        for key in required_keys:
            value = region[key]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"'{key}' must be an integer, got {type(value).__name__}: {value!r}"
                )

        # Validate non-negative for top and left
        if region["top"] < 0:
            raise ValueError(
                f"'top' must be non-negative, got {region['top']}"
            )
        if region["left"] < 0:
            raise ValueError(
                f"'left' must be non-negative, got {region['left']}"
            )

        # Validate width and height >= 1
        if region["width"] < 1:
            raise ValueError(
                f"'width' must be at least 1, got {region['width']}"
            )
        if region["height"] < 1:
            raise ValueError(
                f"'height' must be at least 1, got {region['height']}"
            )

        # Validate within monitor bounds
        monitor_width = monitor["width"]
        monitor_height = monitor["height"]

        if region["left"] + region["width"] > monitor_width:
            raise ValueError(
                f"'left' + 'width' ({region['left'] + region['width']}) exceeds "
                f"monitor width ({monitor_width})"
            )
        if region["top"] + region["height"] > monitor_height:
            raise ValueError(
                f"'top' + 'height' ({region['top'] + region['height']}) exceeds "
                f"monitor height ({monitor_height})"
            )


class DataCollector:
    """Collects training data sequences from screen capture."""

    def __init__(self, actions: list[str], data_path: str = "MP_Data",
                 no_sequences: int = 30, sequence_length: int = 30):
        """
        Args:
            actions: List of action labels (e.g., ["walking", "pointing", "beckoning"]).
            data_path: Root directory for numpy data storage.
            no_sequences: Number of sequences per action (default 30).
            sequence_length: Frames per sequence (default 30).
        """
        self.actions = actions
        self.data_path = data_path
        self.no_sequences = no_sequences
        self.sequence_length = sequence_length

    def create_directories(self) -> None:
        """Create MP_Data/{action}/{sequence}/ folder structure."""
        for action in self.actions:
            for sequence in range(self.no_sequences):
                dir_path = os.path.join(self.data_path, action, str(sequence))
                os.makedirs(dir_path, exist_ok=True)

    def collect(self, screen_capture: ScreenCapture) -> None:
        """
        Run the data collection loop.
        Displays action/sequence info on OpenCV window.
        Pauses 500ms at start of each sequence with "STARTING COLLECTION" message.
        Stops on 'q' key press, preserving completed sequences.
        Retries on frame capture failure without incrementing frame counter.
        """
        # Create directories before collecting
        self.create_directories()

        with mp_holistic.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            stop_requested = False

            for action in self.actions:
                if stop_requested:
                    break

                for sequence in range(self.no_sequences):
                    if stop_requested:
                        break

                    frame_num = 0
                    while frame_num < self.sequence_length:
                        # Try to grab a frame; retry on failure
                        try:
                            frame = screen_capture.grab_frame()
                        except RuntimeError:
                            # 30 consecutive failures - cannot continue
                            logger.error("Screen capture unavailable, stopping collection.")
                            stop_requested = True
                            break
                        except Exception:
                            # Individual frame failure - retry without incrementing
                            continue

                        # Run MediaPipe detection
                        image, results = mediapipe_detection(frame, holistic)

                        # Draw landmarks on the display image
                        draw_styled_landmarks(image, results)

                        # At the start of each sequence, show "STARTING COLLECTION" and pause
                        if frame_num == 0:
                            cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}',
                                        (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(500)
                        else:
                            cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}',
                                        (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', image)

                        # Extract keypoints and save as .npy
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(self.data_path, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        frame_num += 1

                        # Check for 'q' key press to stop
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            stop_requested = True
                            break

        cv2.destroyAllWindows()


class ActionModel:
    """LSTM model for action classification."""

    def __init__(self, actions: list[str], data_path: str = "MP_Data",
                 sequence_length: int = 30, num_keypoints: int = 1662):
        """
        Initialize the ActionModel.

        Args:
            actions: List of action labels (e.g., ["walking", "pointing", "beckoning"]).
            data_path: Root directory for numpy data storage.
            sequence_length: Number of frames per sequence (default 30).
            num_keypoints: Number of keypoint features per frame (default 1662).
        """
        self.actions = actions
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.num_keypoints = num_keypoints
        self.model = self.build_model()

    def build_model(self) -> tf.keras.models.Sequential:
        """
        Build the Sequential LSTM model.

        Architecture:
            LSTM(64, return_sequences=True, relu) →
            LSTM(128, return_sequences=True, relu) →
            LSTM(64, return_sequences=False, relu) →
            Dense(64, relu) → Dense(32, relu) → Dense(num_actions, softmax)

        Input shape: (sequence_length, num_keypoints) i.e. (30, 1662)
        Compiled with: Adam optimizer, categorical_crossentropy, categorical_accuracy.

        Returns:
            Compiled Sequential model.
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(64, return_sequences=True,
                                       activation='relu',
                                       input_shape=(self.sequence_length, self.num_keypoints)))
        model.add(tf.keras.layers.LSTM(128, return_sequences=True,
                                       activation='relu'))
        model.add(tf.keras.layers.LSTM(64, return_sequences=False,
                                       activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(len(self.actions), activation='softmax'))

        model.compile(optimizer='Adam',
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        return model

    def train(self, epochs: int = 2000, test_size: float = 0.05) -> None:
        """
        Train model with stratified 95/5 split.
        Uses TensorBoard callback logging to 'Logs' directory.
        Saves weights to 'screen_action.h5' on completion.

        Args:
            epochs: Number of training epochs (default 2000).
            test_size: Fraction of data for testing (default 0.05 = 5%).
        """
        X, y = self.load_data()

        # Stratified split: use argmax of one-hot labels for stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=np.argmax(y, axis=1)
        )

        # TensorBoard callback logging to 'Logs' directory
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir='Logs')

        # Train the model
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            callbacks=[tb_callback]
        )

        # Save weights on completion
        self.model.save_weights('screen_action.h5')

    def load_weights(self, path: str = "screen_action.h5") -> None:
        """
        Load trained weights. Raises FileNotFoundError if path doesn't exist.

        Args:
            path: Path to the weights file (default 'screen_action.h5').

        Raises:
            FileNotFoundError: If the weights file does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Model weights file not found: '{path}'. "
                f"Train the model first or provide a valid path."
            )
        self.model.load_weights(path)

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load training data from MP_Data/{action}/{sequence}/{frame}.npy.
        Skips sequences with missing/unreadable files (logs warning).
        Raises RuntimeError if fewer than 20 sequences loaded.

        Returns:
            (X, y) where X has shape (N, 30, 1662) and y is one-hot encoded.
        """
        sequences = []
        labels = []

        for action_idx, action in enumerate(self.actions):
            action_path = os.path.join(self.data_path, action)
            if not os.path.isdir(action_path):
                continue

            # Iterate over sequence directories
            for sequence_dir in sorted(os.listdir(action_path)):
                sequence_path = os.path.join(action_path, sequence_dir)
                if not os.path.isdir(sequence_path):
                    continue

                window = []
                skip_sequence = False

                for frame_num in range(self.sequence_length):
                    frame_path = os.path.join(sequence_path, f"{frame_num}.npy")
                    try:
                        frame_data = np.load(frame_path)
                        window.append(frame_data)
                    except Exception as e:
                        logger.warning(
                            f"Skipping sequence: action='{action}', "
                            f"sequence={sequence_dir} - "
                            f"missing/unreadable file '{frame_num}.npy': {e}"
                        )
                        skip_sequence = True
                        break

                if skip_sequence:
                    continue

                if len(window) == self.sequence_length:
                    sequences.append(window)
                    labels.append(action_idx)

        if len(sequences) < 20:
            raise RuntimeError(
                f"Insufficient training data: only {len(sequences)} sequences loaded "
                f"(minimum 20 required)."
            )

        X = np.array(sequences, dtype=np.float64)
        y = tf.keras.utils.to_categorical(labels, num_classes=len(self.actions)).astype(np.float64)

        return X, y


class RealTimeDetector:
    """Real-time action detection from screen capture."""

    def __init__(self, actions: list[str], model_path: str = "screen_action.h5",
                 region: dict | None = None, threshold: float = 0.5):
        """
        Initialize the real-time detector.

        Args:
            actions: Action labels matching training order.
            model_path: Path to trained model weights.
            region: Optional capture region dict with 'top', 'left', 'width', 'height'.
            threshold: Confidence threshold for displaying predictions (default 0.5).

        Raises:
            FileNotFoundError: If model_path does not exist.
        """
        # Validate model file exists
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model weights file not found: '{model_path}'. "
                f"Train the model first or provide a valid path."
            )

        self.actions = actions
        self.threshold = threshold

        # Build and load the model
        self._model = ActionModel(actions=actions)
        self._model.load_weights(model_path)

        # Initialize screen capture
        self._capture = ScreenCapture(region=region)

    def run(self) -> None:
        """
        Main detection loop:
        1. Capture frame from screen
        2. Extract keypoints via MediaPipe Holistic
        3. Maintain sliding window of last 30 keypoint frames
        4. When 30 frames accumulated, run model.predict()
        5. Apply confidence threshold (0.5) and 10-prediction consistency check
        6. Display prob_viz bars and sentence overlay (last 5 actions)
        7. Stop on 'q' key press, release resources
        """
        # State for sliding window and predictions
        sequence = []          # Sliding window of last 30 keypoint frames
        sentence = []          # Last 5 recognized actions for display
        predictions = []       # History of prediction indices for consistency check

        with mp_holistic.Holistic(min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5) as holistic:
            try:
                while True:
                    # 1. Capture frame from screen
                    try:
                        frame = self._capture.grab_frame()
                    except RuntimeError:
                        logger.error("Screen capture unavailable, stopping detection.")
                        break
                    except Exception:
                        # Individual frame failure - skip and retry
                        continue

                    # 2. Extract keypoints via MediaPipe Holistic
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    keypoints = extract_keypoints(results)

                    # 3. Maintain sliding window of last 30 keypoint frames
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    # 4. When 30 frames accumulated, run model.predict()
                    if len(sequence) == 30:
                        input_data = np.expand_dims(sequence, axis=0)
                        res = self._model.model.predict(input_data, verbose=0)[0]

                        # Track prediction index for consistency check
                        predicted_idx = np.argmax(res)
                        predictions.append(predicted_idx)

                        # 5. Apply confidence threshold and 10-prediction consistency check
                        if len(predictions) >= 10:
                            last_10 = predictions[-10:]
                            # Check if last 10 predictions all agree
                            if len(set(last_10)) == 1:
                                predicted_action = self.actions[predicted_idx]
                                confidence = res[predicted_idx]

                                # Only display if confidence >= threshold
                                if confidence >= self.threshold:
                                    # Only append if different from last action in sentence
                                    if len(sentence) == 0 or sentence[-1] != predicted_action:
                                        sentence.append(predicted_action)

                        # Cap sentence at last 5 actions
                        sentence = sentence[-5:]

                        # 6. Display prob_viz bars
                        image = prob_viz(res, self.actions, image, colors)

                    # Display sentence overlay
                    cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                    cv2.putText(image, ' '.join(sentence), (3, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Show the frame
                    cv2.imshow('OpenCV Feed', image)

                    # 7. Stop on 'q' key press
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

            finally:
                # Release resources on exit
                self._capture.release()
                cv2.destroyAllWindows()


def verify_dependencies():
    """Verify all required dependencies are available and importable."""
    dependencies = {
        "mss": mss,
        "numpy": np,
        "cv2 (opencv-python)": cv2,
        "mediapipe": mp,
        "tensorflow": tf,
    }
    print("Dependency verification:")
    for name, module in dependencies.items():
        version = getattr(module, "__version__", "unknown")
        print(f"  {name}: {version}")
    print("  scikit-learn: available")
    print("All dependencies verified successfully.")


def collect_data(region: dict | None = None) -> None:
    """
    Entry point for data collection mode.

    Wires ScreenCapture → DataCollector and runs the collection loop.

    Args:
        region: Optional capture region dict with 'top', 'left', 'width', 'height'.
                If None, captures the entire primary monitor.
    """
    logger.info("Starting data collection mode...")
    capture = ScreenCapture(region=region)
    collector = DataCollector(
        actions=ACTIONS.tolist(),
        data_path=DATA_PATH,
        no_sequences=NO_SEQUENCES,
        sequence_length=SEQUENCE_LENGTH,
    )
    try:
        collector.collect(capture)
    finally:
        capture.release()
    logger.info("Data collection complete.")


def train_model(epochs: int = 2000) -> None:
    """
    Entry point for model training mode.

    Creates an ActionModel, loads data, and trains the model.

    Args:
        epochs: Number of training epochs (default 2000).
    """
    logger.info("Starting model training mode...")
    model = ActionModel(
        actions=ACTIONS.tolist(),
        data_path=DATA_PATH,
        sequence_length=SEQUENCE_LENGTH,
        num_keypoints=NUM_KEYPOINTS,
    )
    model.train(epochs=epochs)
    logger.info("Model training complete. Weights saved to '%s'.", MODEL_PATH)


def detect_actions(region: dict | None = None) -> None:
    """
    Entry point for real-time detection mode.

    Creates a RealTimeDetector and runs the detection loop.

    Args:
        region: Optional capture region dict with 'top', 'left', 'width', 'height'.
                If None, captures the entire primary monitor.
    """
    logger.info("Starting real-time detection mode...")
    detector = RealTimeDetector(
        actions=ACTIONS.tolist(),
        model_path=MODEL_PATH,
        region=region,
    )
    detector.run()
    logger.info("Detection stopped.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Screen Action Recognition System - detect actions in videos playing on screen."
    )
    parser.add_argument(
        "mode",
        choices=["collect", "train", "detect"],
        help="Operation mode: 'collect' to gather training data, "
             "'train' to train the LSTM model, "
             "'detect' to run real-time action detection.",
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Top coordinate of the capture region (pixels).",
    )
    parser.add_argument(
        "--left", type=int, default=None,
        help="Left coordinate of the capture region (pixels).",
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Width of the capture region (pixels).",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Height of the capture region (pixels).",
    )
    parser.add_argument(
        "--epochs", type=int, default=2000,
        help="Number of training epochs (only used in 'train' mode, default 2000).",
    )

    args = parser.parse_args()

    # Build region dict if any region argument is provided
    region = None
    region_args = [args.top, args.left, args.width, args.height]
    if any(v is not None for v in region_args):
        # If any region arg is specified, all must be specified
        if not all(v is not None for v in region_args):
            parser.error(
                "All region arguments (--top, --left, --width, --height) "
                "must be provided together."
            )
        region = {
            "top": args.top,
            "left": args.left,
            "width": args.width,
            "height": args.height,
        }

    if args.mode == "collect":
        collect_data(region=region)
    elif args.mode == "train":
        train_model(epochs=args.epochs)
    elif args.mode == "detect":
        detect_actions(region=region)
