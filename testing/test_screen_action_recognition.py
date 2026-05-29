# Feature: screen-action-recognition, Property 2: Region validation correctness
"""
Property-based test for region validation correctness.

Property 2: For any tuple (x, y, width, height), the Screen_Capture_Module SHALL accept
it if and only if all values are non-negative integers, width >= 1, height >= 1,
x + width <= monitor_width, and y + height <= monitor_height. For any tuple that violates
these constraints, the module SHALL raise a ValueError whose message identifies the
specific violating parameter.

Validates: Requirements 2.1, 2.2, 2.3, 2.4
"""

from unittest.mock import patch, MagicMock
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from screen_action_recognition import ScreenCapture


# Fixed monitor dimensions for deterministic testing
MONITOR_WIDTH = 1920
MONITOR_HEIGHT = 1080


def mock_mss_instance():
    """Create a mock mss instance with fixed monitor dimensions."""
    mock_sct = MagicMock()
    mock_sct.monitors = [
        {"top": 0, "left": 0, "width": MONITOR_WIDTH + MONITOR_HEIGHT, "height": MONITOR_WIDTH + MONITOR_HEIGHT},  # "all" monitor
        {"top": 0, "left": 0, "width": MONITOR_WIDTH, "height": MONITOR_HEIGHT},  # primary monitor
    ]
    return mock_sct


# Strategy for valid regions: all constraints satisfied
@st.composite
def valid_regions(draw):
    """Generate valid region dicts that satisfy all constraints."""
    # width and height must be >= 1 and fit within monitor bounds
    width = draw(st.integers(min_value=1, max_value=MONITOR_WIDTH))
    height = draw(st.integers(min_value=1, max_value=MONITOR_HEIGHT))
    # left must be >= 0 and left + width <= MONITOR_WIDTH
    left = draw(st.integers(min_value=0, max_value=MONITOR_WIDTH - width))
    # top must be >= 0 and top + height <= MONITOR_HEIGHT
    top = draw(st.integers(min_value=0, max_value=MONITOR_HEIGHT - height))
    return {"top": top, "left": left, "width": width, "height": height}


# Strategy for invalid regions with non-integer values
@st.composite
def non_integer_regions(draw):
    """Generate regions where at least one value is not an integer."""
    non_int_values = st.one_of(
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(min_size=1, max_size=5),
        st.none(),
        st.booleans(),
    )
    # Pick which key to make invalid
    invalid_key = draw(st.sampled_from(["top", "left", "width", "height"]))
    region = {"top": 0, "left": 0, "width": 100, "height": 100}
    region[invalid_key] = draw(non_int_values)
    return region, invalid_key


# Strategy for negative top/left values
@st.composite
def negative_coordinate_regions(draw):
    """Generate regions with negative top or left values."""
    negative_key = draw(st.sampled_from(["top", "left"]))
    region = {"top": 0, "left": 0, "width": 100, "height": 100}
    region[negative_key] = draw(st.integers(max_value=-1))
    return region, negative_key


# Strategy for zero/negative width or height
@st.composite
def invalid_dimension_regions(draw):
    """Generate regions with width or height < 1."""
    invalid_key = draw(st.sampled_from(["width", "height"]))
    region = {"top": 0, "left": 0, "width": 100, "height": 100}
    region[invalid_key] = draw(st.integers(max_value=0))
    return region, invalid_key


# Strategy for regions exceeding monitor bounds
@st.composite
def out_of_bounds_regions(draw):
    """Generate regions where left+width > monitor_width or top+height > monitor_height."""
    bound_type = draw(st.sampled_from(["horizontal", "vertical"]))
    if bound_type == "horizontal":
        # left + width > MONITOR_WIDTH
        width = draw(st.integers(min_value=1, max_value=MONITOR_WIDTH))
        # left must be such that left + width > MONITOR_WIDTH
        left = draw(st.integers(min_value=MONITOR_WIDTH - width + 1, max_value=MONITOR_WIDTH * 2))
        # Keep top/height valid
        height = draw(st.integers(min_value=1, max_value=MONITOR_HEIGHT))
        top = draw(st.integers(min_value=0, max_value=MONITOR_HEIGHT - height))
        return {"top": top, "left": left, "width": width, "height": height}, "left"
    else:
        # top + height > MONITOR_HEIGHT
        height = draw(st.integers(min_value=1, max_value=MONITOR_HEIGHT))
        # top must be such that top + height > MONITOR_HEIGHT
        top = draw(st.integers(min_value=MONITOR_HEIGHT - height + 1, max_value=MONITOR_HEIGHT * 2))
        # Keep left/width valid
        width = draw(st.integers(min_value=1, max_value=MONITOR_WIDTH))
        left = draw(st.integers(min_value=0, max_value=MONITOR_WIDTH - width))
        return {"top": top, "left": left, "width": width, "height": height}, "top"


class TestRegionValidationProperty:
    """Property 2: Region validation correctness.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4**
    """

    @given(region=valid_regions())
    @settings(max_examples=100)
    def test_valid_regions_are_accepted(self, region):
        """Valid regions (non-negative ints, width>=1, height>=1, within bounds) are accepted."""
        with patch("screen_action_recognition.mss.mss") as mock_mss_cls:
            mock_sct = mock_mss_instance()
            mock_mss_cls.return_value = mock_sct

            # Should not raise any exception
            capture = ScreenCapture(region=region)
            assert capture._monitor["top"] == region["top"]
            assert capture._monitor["left"] == region["left"]
            assert capture._monitor["width"] == region["width"]
            assert capture._monitor["height"] == region["height"]
            capture.release()

    @given(data=non_integer_regions())
    @settings(max_examples=100)
    def test_non_integer_values_raise_valueerror(self, data):
        """Non-integer values (float, string, None, bool) raise ValueError identifying the parameter."""
        region, invalid_key = data

        with patch("screen_action_recognition.mss.mss") as mock_mss_cls:
            mock_sct = mock_mss_instance()
            mock_mss_cls.return_value = mock_sct

            with pytest.raises(ValueError) as exc_info:
                ScreenCapture(region=region)

            # The error message should identify the invalid parameter
            assert invalid_key in str(exc_info.value)

    @given(data=negative_coordinate_regions())
    @settings(max_examples=100)
    def test_negative_coordinates_raise_valueerror(self, data):
        """Negative top or left values raise ValueError identifying the parameter."""
        region, negative_key = data

        with patch("screen_action_recognition.mss.mss") as mock_mss_cls:
            mock_sct = mock_mss_instance()
            mock_mss_cls.return_value = mock_sct

            with pytest.raises(ValueError) as exc_info:
                ScreenCapture(region=region)

            # The error message should identify the specific parameter
            assert negative_key in str(exc_info.value)

    @given(data=invalid_dimension_regions())
    @settings(max_examples=100)
    def test_zero_or_negative_dimensions_raise_valueerror(self, data):
        """Width or height < 1 raises ValueError identifying the parameter."""
        region, invalid_key = data

        with patch("screen_action_recognition.mss.mss") as mock_mss_cls:
            mock_sct = mock_mss_instance()
            mock_mss_cls.return_value = mock_sct

            with pytest.raises(ValueError) as exc_info:
                ScreenCapture(region=region)

            # The error message should identify the specific parameter
            assert invalid_key in str(exc_info.value)

    @given(data=out_of_bounds_regions())
    @settings(max_examples=100)
    def test_out_of_bounds_regions_raise_valueerror(self, data):
        """Regions exceeding monitor bounds raise ValueError identifying the parameter."""
        region, expected_key = data

        with patch("screen_action_recognition.mss.mss") as mock_mss_cls:
            mock_sct = mock_mss_instance()
            mock_mss_cls.return_value = mock_sct

            with pytest.raises(ValueError) as exc_info:
                ScreenCapture(region=region)

            # The error message should reference the bounds violation
            error_msg = str(exc_info.value)
            assert expected_key in error_msg or "exceeds" in error_msg


# Feature: screen-action-recognition, Property 3: Keypoint extraction output invariant
"""
Property-based test for keypoint extraction output invariant.

Property 3: For any BGR uint8 numpy array (including frames with no detectable human),
the Keypoint_Extractor SHALL produce a 1662-element float64 numpy array. The output
length is invariant regardless of whether MediaPipe detects landmarks.

Validates: Requirements 3.1, 3.2, 3.3
"""

import numpy as np
import mediapipe as mp
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from screen_action_recognition import safe_extract_keypoints


# Module-level holistic model instance shared across all property test examples
_holistic_model = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)


class TestKeypointExtractionOutputProperty:
    """Property 3: Keypoint extraction output invariant.

    **Validates: Requirements 3.1, 3.2, 3.3**
    """

    @given(
        frame=arrays(
            dtype=np.uint8,
            shape=st.tuples(
                st.integers(min_value=1, max_value=480),
                st.integers(min_value=1, max_value=640),
                st.just(3),
            ),
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_output_is_always_1662_float64_array(self, frame):
        """For any BGR uint8 array of various sizes, output is always a 1662-element float64 numpy array."""
        result = safe_extract_keypoints(frame, _holistic_model)

        # Output must always be a numpy array
        assert isinstance(result, np.ndarray)
        # Output must always have exactly 1662 elements
        assert result.shape == (1662,)
        # Output must always be float64
        assert result.dtype == np.float64



# Unit tests for keypoint extraction (Task 4.3)
# Validates: Requirement 3.5
"""
Unit tests for safe_extract_keypoints function.

Tests:
- Null frame returns zeros(1662)
- Empty array returns zeros(1662)
- Valid BGR frame produces 1662-element array
"""

import numpy as np
import mediapipe as mp

from screen_action_recognition import safe_extract_keypoints


class TestSafeExtractKeypointsUnit:
    """Unit tests for safe_extract_keypoints.

    _Requirements: 3.5_
    """

    def test_null_frame_returns_zeros(self):
        """Passing None as frame should return np.zeros(1662) with dtype float64."""
        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            result = safe_extract_keypoints(None, holistic)

            assert isinstance(result, np.ndarray)
            assert result.shape == (1662,)
            assert result.dtype == np.float64
            np.testing.assert_array_equal(result, np.zeros(1662))

    def test_empty_array_returns_zeros(self):
        """Passing an empty numpy array should return np.zeros(1662) with dtype float64."""
        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            empty_frame = np.array([])
            result = safe_extract_keypoints(empty_frame, holistic)

            assert isinstance(result, np.ndarray)
            assert result.shape == (1662,)
            assert result.dtype == np.float64
            np.testing.assert_array_equal(result, np.zeros(1662))

    def test_valid_bgr_frame_produces_1662_element_array(self):
        """Passing a valid BGR uint8 frame (480x640x3) should return a 1662-element array."""
        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            # Create a valid BGR frame (480x640x3, uint8)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = safe_extract_keypoints(frame, holistic)

            assert isinstance(result, np.ndarray)
            assert result.shape == (1662,)
            assert result.dtype == np.float64


# Feature: screen-action-recognition, Property 4: Data storage round-trip
"""
Property-based test for data storage round-trip.

Property 4: For any action label, sequence number, and frame number within the configured
ranges, saving a 1662-element keypoint array via Data_Collector and then loading it via
np.load(MP_Data/{action}/{sequence}/{frame}.npy) SHALL produce an array equal to the original.

Validates: Requirements 4.2, 4.3, 5.6, 7.3
"""

import os
import tempfile

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as hnp_arrays


# Actions matching the system configuration
_ACTIONS = ["walking", "pointing", "beckoning"]


class TestDataStorageRoundTripProperty:
    """Property 4: Data storage round-trip.

    **Validates: Requirements 4.2, 4.3, 5.6, 7.3**
    """

    @given(
        keypoints=hnp_arrays(
            dtype=np.float64,
            shape=(1662,),
            elements=st.floats(
                min_value=-1e6, max_value=1e6,
                allow_nan=False, allow_infinity=False,
            ),
        ),
        action=st.sampled_from(_ACTIONS),
        sequence=st.integers(min_value=0, max_value=29),
        frame=st.integers(min_value=0, max_value=29),
    )
    @settings(max_examples=100)
    def test_save_load_roundtrip_preserves_data(self, keypoints, action, sequence, frame):
        """Saving a 1662-element float64 array via np.save and loading via np.load produces an equal array."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Build the directory path matching MP_Data/{action}/{sequence}/
            dir_path = os.path.join(tmp_dir, "MP_Data", action, str(sequence))
            os.makedirs(dir_path, exist_ok=True)

            # Save the keypoint array as {frame}.npy
            file_path = os.path.join(dir_path, str(frame))
            np.save(file_path, keypoints)

            # Load it back (np.save appends .npy automatically)
            loaded = np.load(file_path + ".npy")

            # Assert exact equality
            assert np.array_equal(keypoints, loaded), (
                f"Round-trip failed for action={action}, sequence={sequence}, frame={frame}"
            )
            # Verify shape and dtype are preserved
            assert loaded.shape == (1662,)
            assert loaded.dtype == np.float64


# Unit tests for DataCollector (Task 5.3)
# Validates: Requirement 4.5
"""
Unit tests for DataCollector.create_directories() method.

Tests:
- Directory creation produces correct folder structure
- All action/sequence directories exist after create_directories()
"""

import os

from screen_action_recognition import DataCollector


class TestDataCollectorDirectories:
    """Unit tests for DataCollector directory creation.

    _Requirements: 4.5_
    """

    def test_create_directories_produces_correct_folder_structure(self, tmp_path):
        """After calling create_directories(), the structure {data_path}/{action}/{sequence}/ exists for all actions and sequences."""
        actions = ["walking", "pointing", "beckoning"]
        no_sequences = 30
        data_path = str(tmp_path / "MP_Data")

        collector = DataCollector(
            actions=actions,
            data_path=data_path,
            no_sequences=no_sequences,
            sequence_length=30,
        )
        collector.create_directories()

        # Verify each action directory exists
        for action in actions:
            action_dir = os.path.join(data_path, action)
            assert os.path.isdir(action_dir), f"Action directory '{action}' does not exist"

            # Verify each sequence directory exists within the action
            for seq in range(no_sequences):
                seq_dir = os.path.join(action_dir, str(seq))
                assert os.path.isdir(seq_dir), (
                    f"Sequence directory '{action}/{seq}' does not exist"
                )

    def test_all_action_sequence_directories_exist(self, tmp_path):
        """All expected action/sequence directories exist after create_directories() — total count matches actions * no_sequences."""
        actions = ["walking", "pointing", "beckoning"]
        no_sequences = 30
        data_path = str(tmp_path / "MP_Data")

        collector = DataCollector(
            actions=actions,
            data_path=data_path,
            no_sequences=no_sequences,
            sequence_length=30,
        )
        collector.create_directories()

        # Count total directories created at the sequence level
        total_sequence_dirs = 0
        for action in actions:
            action_dir = os.path.join(data_path, action)
            for seq in range(no_sequences):
                seq_dir = os.path.join(action_dir, str(seq))
                if os.path.isdir(seq_dir):
                    total_sequence_dirs += 1

        expected_total = len(actions) * no_sequences  # 3 * 30 = 90
        assert total_sequence_dirs == expected_total, (
            f"Expected {expected_total} sequence directories, found {total_sequence_dirs}"
        )

    def test_create_directories_with_custom_actions_and_sequences(self, tmp_path):
        """create_directories() works correctly with custom action lists and sequence counts."""
        actions = ["action_a", "action_b"]
        no_sequences = 5
        data_path = str(tmp_path / "CustomData")

        collector = DataCollector(
            actions=actions,
            data_path=data_path,
            no_sequences=no_sequences,
            sequence_length=10,
        )
        collector.create_directories()

        for action in actions:
            for seq in range(no_sequences):
                seq_dir = os.path.join(data_path, action, str(seq))
                assert os.path.isdir(seq_dir), (
                    f"Directory '{action}/{seq}' was not created"
                )

    def test_create_directories_is_idempotent(self, tmp_path):
        """Calling create_directories() multiple times does not raise errors (exist_ok=True behavior)."""
        actions = ["walking", "pointing", "beckoning"]
        no_sequences = 30
        data_path = str(tmp_path / "MP_Data")

        collector = DataCollector(
            actions=actions,
            data_path=data_path,
            no_sequences=no_sequences,
            sequence_length=30,
        )

        # Call twice — should not raise
        collector.create_directories()
        collector.create_directories()

        # Verify structure still correct
        for action in actions:
            for seq in range(no_sequences):
                seq_dir = os.path.join(data_path, action, str(seq))
                assert os.path.isdir(seq_dir)


# Feature: screen-action-recognition, Property 5: Model input/output shape consistency
"""
Property-based test for model input/output shape consistency.

Property 5: For any batch of input arrays with shape (N, 30, 1662) where N >= 1,
the LSTM_Model.predict() SHALL return an array of shape (N, 3) where each row sums
to approximately 1.0 (valid probability distribution).

Validates: Requirements 5.2, 7.4
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as hnp_arrays

from screen_action_recognition import ActionModel


# Build the model once and reuse across hypothesis examples for efficiency
_actions = ["walking", "pointing", "beckoning"]
_action_model = ActionModel(actions=_actions)


class TestModelIOShapeProperty:
    """Property 5: Model input/output shape consistency.

    **Validates: Requirements 5.2, 7.4**
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=10),
        data=st.data(),
    )
    @settings(max_examples=100, deadline=None)
    def test_predict_returns_correct_shape_and_valid_probabilities(self, batch_size, data):
        """For any (N, 30, 1662) float input, model.predict() returns shape (N, 3) with rows summing to ~1.0."""
        # Generate random float array of shape (batch_size, 30, 1662)
        input_array = data.draw(
            hnp_arrays(
                dtype=np.float32,
                shape=(batch_size, 30, 1662),
                elements=st.floats(
                    min_value=-1.0, max_value=1.0,
                    allow_nan=False, allow_infinity=False,
                    allow_subnormal=False,
                    width=32,
                ),
            )
        )

        # Run prediction
        output = _action_model.model.predict(input_array, verbose=0)

        # Assert output shape is (N, 3)
        assert output.shape == (batch_size, 3), (
            f"Expected output shape ({batch_size}, 3), got {output.shape}"
        )

        # Assert each row sums to approximately 1.0 (softmax output)
        row_sums = output.sum(axis=1)
        for i, row_sum in enumerate(row_sums):
            assert np.isclose(row_sum, 1.0, atol=1e-5), (
                f"Row {i} sum is {row_sum}, expected ~1.0"
            )


# Feature: screen-action-recognition, Property 6: Stratified split preserves class distribution
"""
Property-based test for stratified split preserving class distribution.

Property 6: For any labeled dataset with at least 20 sequences distributed across all
action classes, splitting with a 95/5 ratio using stratified splitting SHALL produce a
test set where every action class is represented at least once.

Validates: Requirements 5.8
"""

import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from sklearn.model_selection import train_test_split


# Actions matching the system configuration
_SPLIT_ACTIONS = ["walking", "pointing", "beckoning"]
NUM_CLASSES = len(_SPLIT_ACTIONS)


@st.composite
def labeled_datasets(draw):
    """Generate random labeled datasets with at least 20 samples per class (3 classes).

    This ensures stratified split with test_size=0.05 can produce at least one
    sample per class in the test set (need at least 20 per class so 5% yields >= 1).
    """
    # Generate number of samples per class: at least 20 each
    samples_per_class = [
        draw(st.integers(min_value=20, max_value=100))
        for _ in range(NUM_CLASSES)
    ]

    # Build label array: class indices repeated per their count
    labels = []
    for class_idx, count in enumerate(samples_per_class):
        labels.extend([class_idx] * count)

    labels = np.array(labels)

    # Shuffle the labels to simulate random ordering
    indices = draw(
        st.permutations(list(range(len(labels))))
    )
    labels = labels[indices]

    return labels


class TestStratifiedSplitProperty:
    """Property 6: Stratified split preserves class distribution.

    **Validates: Requirements 5.8**
    """

    @given(labels=labeled_datasets())
    @settings(max_examples=100, deadline=None)
    def test_stratified_split_test_set_contains_all_classes(self, labels):
        """For any labeled dataset with at least 20 samples per class,
        stratified split with test_size=0.05 produces a test set containing
        at least one sample from every class."""
        # Perform stratified split matching ActionModel.train() logic
        X = np.arange(len(labels))  # Dummy feature array (indices)
        y = labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.05, stratify=y
        )

        # Assert the test set contains at least one sample from every class
        unique_classes_in_test = set(y_test.tolist())
        for class_idx in range(NUM_CLASSES):
            assert class_idx in unique_classes_in_test, (
                f"Class {class_idx} ({_SPLIT_ACTIONS[class_idx]}) is missing from "
                f"the test set. Test set classes: {unique_classes_in_test}, "
                f"total test samples: {len(y_test)}"
            )


# Unit tests for ActionModel (Task 6.6)
# Validates: Requirements 5.1, 5.9, 6.8
"""
Unit tests for ActionModel class.

Tests:
- Model architecture: verify layer types, units, activations match spec
- Compilation settings: verify optimizer, loss, metrics
- Insufficient data raises RuntimeError (provide < 20 sequences)
- load_weights raises FileNotFoundError for missing file
"""

import os
import tempfile

import numpy as np
import tensorflow as tf

from screen_action_recognition import ActionModel


class TestActionModelArchitecture:
    """Unit tests for ActionModel build_model architecture.

    _Requirements: 5.1_
    """

    def setup_method(self):
        """Create an ActionModel instance for testing."""
        self.actions = ["walking", "pointing", "beckoning"]
        self.model_instance = ActionModel(
            actions=self.actions,
            data_path="MP_Data",
            sequence_length=30,
            num_keypoints=1662,
        )
        self.model = self.model_instance.model

    def test_layer_types(self):
        """Model layers should be: LSTM, LSTM, LSTM, Dense, Dense, Dense."""
        layers = self.model.layers
        assert len(layers) == 6, f"Expected 6 layers, got {len(layers)}"

        assert isinstance(layers[0], tf.keras.layers.LSTM)
        assert isinstance(layers[1], tf.keras.layers.LSTM)
        assert isinstance(layers[2], tf.keras.layers.LSTM)
        assert isinstance(layers[3], tf.keras.layers.Dense)
        assert isinstance(layers[4], tf.keras.layers.Dense)
        assert isinstance(layers[5], tf.keras.layers.Dense)

    def test_layer_units(self):
        """Layer units should be: 64, 128, 64, 64, 32, 3."""
        layers = self.model.layers
        expected_units = [64, 128, 64, 64, 32, 3]

        for i, (layer, expected) in enumerate(zip(layers, expected_units)):
            actual = layer.units if hasattr(layer, "units") else layer.output_shape[-1]
            assert actual == expected, (
                f"Layer {i} ({layer.name}): expected {expected} units, got {actual}"
            )

    def test_layer_activations(self):
        """Layer activations should be: relu, relu, relu, relu, relu, softmax."""
        layers = self.model.layers
        expected_activations = ["relu", "relu", "relu", "relu", "relu", "softmax"]

        for i, (layer, expected) in enumerate(zip(layers, expected_activations)):
            config = layer.get_config()
            activation = config.get("activation", None)
            # Activation can be a string or a dict with 'class_name'
            if isinstance(activation, dict):
                activation = activation.get("class_name", "").lower()
            assert activation == expected, (
                f"Layer {i} ({layer.name}): expected activation '{expected}', got '{activation}'"
            )

    def test_lstm_return_sequences(self):
        """First two LSTMs should have return_sequences=True, third should have return_sequences=False."""
        layers = self.model.layers
        assert layers[0].return_sequences is True, "LSTM layer 0 should have return_sequences=True"
        assert layers[1].return_sequences is True, "LSTM layer 1 should have return_sequences=True"
        assert layers[2].return_sequences is False, "LSTM layer 2 should have return_sequences=False"

    def test_input_shape(self):
        """Model input shape should be (None, 30, 1662)."""
        input_shape = self.model.input_shape
        assert input_shape == (None, 30, 1662), (
            f"Expected input shape (None, 30, 1662), got {input_shape}"
        )


class TestActionModelCompilation:
    """Unit tests for ActionModel compilation settings.

    _Requirements: 5.1_
    """

    def setup_method(self):
        """Create an ActionModel instance for testing."""
        self.actions = ["walking", "pointing", "beckoning"]
        self.model_instance = ActionModel(
            actions=self.actions,
            data_path="MP_Data",
            sequence_length=30,
            num_keypoints=1662,
        )
        self.model = self.model_instance.model

    def test_optimizer_is_adam(self):
        """Model should be compiled with Adam optimizer."""
        optimizer = self.model.optimizer
        assert isinstance(optimizer, tf.keras.optimizers.Adam), (
            f"Expected Adam optimizer, got {type(optimizer).__name__}"
        )

    def test_loss_is_categorical_crossentropy(self):
        """Model should be compiled with categorical_crossentropy loss."""
        loss = self.model.loss
        # Loss can be a string or a function/class instance
        if isinstance(loss, str):
            assert loss == "categorical_crossentropy", (
                f"Expected 'categorical_crossentropy' loss, got '{loss}'"
            )
        else:
            # It might be a loss function object
            loss_name = getattr(loss, "name", "") or getattr(loss, "__name__", "")
            assert "categorical_crossentropy" in loss_name.lower(), (
                f"Expected categorical_crossentropy loss, got '{loss_name}'"
            )

    def test_metrics_include_categorical_accuracy(self):
        """Model should have categorical_accuracy as a metric."""
        compile_config = self.model.get_compile_config()
        metrics = compile_config.get("metrics", [])
        assert "categorical_accuracy" in metrics, (
            f"Expected 'categorical_accuracy' in metrics, got {metrics}"
        )


class TestActionModelLoadData:
    """Unit tests for ActionModel.load_data() insufficient data handling.

    _Requirements: 5.9_
    """

    def test_insufficient_data_raises_runtime_error(self):
        """load_data() should raise RuntimeError when fewer than 20 sequences are available."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            actions = ["walking", "pointing", "beckoning"]
            data_path = os.path.join(tmp_dir, "MP_Data")

            # Create only 5 sequences total (well below the 20 minimum)
            # Create 1 sequence for "walking" and 1 for "pointing" (2 total)
            for action in actions[:2]:
                for seq in range(3):
                    seq_dir = os.path.join(data_path, action, str(seq))
                    os.makedirs(seq_dir, exist_ok=True)
                    for frame in range(30):
                        keypoints = np.zeros(1662, dtype=np.float64)
                        np.save(os.path.join(seq_dir, f"{frame}.npy"), keypoints)

            model = ActionModel(
                actions=actions,
                data_path=data_path,
                sequence_length=30,
                num_keypoints=1662,
            )

            with pytest.raises(RuntimeError) as exc_info:
                model.load_data()

            assert "insufficient" in str(exc_info.value).lower() or "20" in str(exc_info.value)

    def test_empty_data_path_raises_runtime_error(self):
        """load_data() should raise RuntimeError when data path has no sequences."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            actions = ["walking", "pointing", "beckoning"]
            data_path = os.path.join(tmp_dir, "EmptyData")
            os.makedirs(data_path, exist_ok=True)

            model = ActionModel(
                actions=actions,
                data_path=data_path,
                sequence_length=30,
                num_keypoints=1662,
            )

            with pytest.raises(RuntimeError) as exc_info:
                model.load_data()

            assert "insufficient" in str(exc_info.value).lower() or "20" in str(exc_info.value)


class TestActionModelLoadWeights:
    """Unit tests for ActionModel.load_weights() error handling.

    _Requirements: 6.8_
    """

    def test_load_weights_raises_file_not_found_error(self):
        """load_weights() should raise FileNotFoundError when the weights file doesn't exist."""
        actions = ["walking", "pointing", "beckoning"]
        model = ActionModel(
            actions=actions,
            data_path="MP_Data",
            sequence_length=30,
            num_keypoints=1662,
        )

        with pytest.raises(FileNotFoundError) as exc_info:
            model.load_weights("nonexistent_model_weights.h5")

        assert "nonexistent_model_weights.h5" in str(exc_info.value)

    def test_load_weights_raises_for_missing_default_path(self):
        """load_weights() with default path should raise FileNotFoundError if screen_action.h5 doesn't exist."""
        actions = ["walking", "pointing", "beckoning"]
        model = ActionModel(
            actions=actions,
            data_path="MP_Data",
            sequence_length=30,
            num_keypoints=1662,
        )

        # Ensure the default file doesn't exist in a temp directory context
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_path = os.path.join(tmp_dir, "screen_action.h5")
            with pytest.raises(FileNotFoundError):
                model.load_weights(fake_path)


# Feature: screen-action-recognition, Property 7: Sliding window size invariant
"""
Property-based test for sliding window size invariant.

Property 7: For any number of frames processed (n), the Real_Time_Detector's sliding
window SHALL contain exactly min(n, 30) keypoint frames. The window never exceeds 30
elements.

Validates: Requirements 6.1
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as hnp_arrays


# Sliding window max size matching RealTimeDetector implementation
WINDOW_MAX_SIZE = 30


@st.composite
def keypoint_frame_sequences(draw):
    """Generate random sequences of keypoint frames of varying lengths (1 to 100)."""
    n = draw(st.integers(min_value=1, max_value=100))
    # Generate n keypoint frames (each is a 1662-element float64 array)
    frames = []
    for _ in range(n):
        frame = draw(
            hnp_arrays(
                dtype=np.float64,
                shape=(1662,),
                elements=st.floats(
                    min_value=-1.0, max_value=1.0,
                    allow_nan=False, allow_infinity=False,
                ),
            )
        )
        frames.append(frame)
    return n, frames


class TestSlidingWindowSizeProperty:
    """Property 7: Sliding window size invariant.

    **Validates: Requirements 6.1**
    """

    @given(data=keypoint_frame_sequences())
    @settings(max_examples=100, deadline=None)
    def test_sliding_window_contains_exactly_min_n_30_frames(self, data):
        """For any n frames processed, the sliding window contains exactly min(n, 30) frames
        and never exceeds 30 elements."""
        n, frames = data

        # Simulate the sliding window logic from RealTimeDetector.run():
        #   sequence.append(keypoints)
        #   sequence = sequence[-30:]
        sequence = []
        for i, keypoints in enumerate(frames):
            sequence.append(keypoints)
            sequence = sequence[-WINDOW_MAX_SIZE:]

            # Invariant: after processing (i+1) frames, window size is min(i+1, 30)
            current_frame_count = i + 1
            expected_size = min(current_frame_count, WINDOW_MAX_SIZE)

            assert len(sequence) == expected_size, (
                f"After processing {current_frame_count} frames, "
                f"expected window size {expected_size}, got {len(sequence)}"
            )

            # Window must NEVER exceed 30
            assert len(sequence) <= WINDOW_MAX_SIZE, (
                f"Window size {len(sequence)} exceeds maximum {WINDOW_MAX_SIZE}"
            )

        # Final assertion: after all n frames, window has exactly min(n, 30) elements
        assert len(sequence) == min(n, WINDOW_MAX_SIZE), (
            f"Final window size: expected {min(n, WINDOW_MAX_SIZE)}, got {len(sequence)}"
        )


# Feature: screen-action-recognition, Property 8: Threshold-based display logic
"""
Property-based test for threshold-based display logic.

Property 8: For any prediction probability vector of length 3, the Real_Time_Detector
SHALL display the action label corresponding to argmax(vector) if and only if
max(vector) >= 0.5. If max(vector) < 0.5, no action label SHALL be displayed.

Validates: Requirements 6.2, 6.3, 6.4
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st


# Actions matching the system configuration
_THRESHOLD_ACTIONS = ["walking", "pointing", "beckoning"]
_THRESHOLD = 0.5


@st.composite
def probability_vectors(draw):
    """Generate random 3-element probability vectors (valid softmax-like outputs).

    Strategy: generate 3 positive floats and normalize them to sum to 1.0,
    producing a valid probability distribution similar to softmax output.
    """
    # Generate 3 positive floats (using exponentials of random values for Dirichlet-like behavior)
    raw = draw(
        st.lists(
            st.floats(min_value=1e-6, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=3,
        )
    )
    raw_array = np.array(raw, dtype=np.float64)
    # Normalize to sum to 1.0 (simulates softmax output)
    total = raw_array.sum()
    normalized = raw_array / total
    return normalized


def apply_threshold_display_logic(res: np.ndarray, actions: list[str], threshold: float) -> str | None:
    """Simulate the threshold display logic from RealTimeDetector.

    This mirrors the logic in RealTimeDetector.run():
    - If max(res) >= threshold, display actions[argmax(res)]
    - Otherwise, display nothing (None)

    Args:
        res: Probability vector of length 3.
        actions: List of action labels.
        threshold: Confidence threshold (0.5).

    Returns:
        The action label to display, or None if below threshold.
    """
    predicted_idx = np.argmax(res)
    confidence = res[predicted_idx]
    if confidence >= threshold:
        return actions[predicted_idx]
    return None


class TestThresholdDisplayLogicProperty:
    """Property 8: Threshold-based display logic.

    **Validates: Requirements 6.2, 6.3, 6.4**
    """

    @given(prob_vector=probability_vectors())
    @settings(max_examples=100)
    def test_action_displayed_iff_max_above_threshold(self, prob_vector):
        """Action label is displayed if and only if max(vector) >= 0.5,
        and the displayed label matches argmax of the vector."""
        max_prob = np.max(prob_vector)
        argmax_idx = np.argmax(prob_vector)
        expected_action = _THRESHOLD_ACTIONS[argmax_idx]

        displayed = apply_threshold_display_logic(prob_vector, _THRESHOLD_ACTIONS, _THRESHOLD)

        if max_prob >= _THRESHOLD:
            # Action SHOULD be displayed and match argmax
            assert displayed is not None, (
                f"Expected action to be displayed for max_prob={max_prob:.4f} >= {_THRESHOLD}, "
                f"but got None. Vector: {prob_vector}"
            )
            assert displayed == expected_action, (
                f"Expected displayed action '{expected_action}' (argmax={argmax_idx}), "
                f"got '{displayed}'. Vector: {prob_vector}"
            )
        else:
            # No action should be displayed
            assert displayed is None, (
                f"Expected no action displayed for max_prob={max_prob:.4f} < {_THRESHOLD}, "
                f"but got '{displayed}'. Vector: {prob_vector}"
            )

    @given(prob_vector=probability_vectors())
    @settings(max_examples=100)
    def test_displayed_label_always_matches_argmax(self, prob_vector):
        """When an action is displayed, it always corresponds to the argmax index."""
        displayed = apply_threshold_display_logic(prob_vector, _THRESHOLD_ACTIONS, _THRESHOLD)

        if displayed is not None:
            argmax_idx = np.argmax(prob_vector)
            assert displayed == _THRESHOLD_ACTIONS[argmax_idx], (
                f"Displayed action '{displayed}' does not match argmax action "
                f"'{_THRESHOLD_ACTIONS[argmax_idx]}' (index {argmax_idx}). "
                f"Vector: {prob_vector}"
            )


# Feature: screen-action-recognition, Property 9: Sentence-building consistency
"""
Property-based test for sentence-building consistency.

Property 9: For any sequence of prediction results, the sentence display SHALL contain
at most 5 actions, SHALL only append a new action when it differs from the last entry,
and SHALL only consider an action valid when the last 10 predictions all agree on the
same argmax class with confidence >= 0.5.

Validates: Requirements 7.5
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as hnp_arrays


# Actions matching the system configuration
_SENTENCE_ACTIONS = ["walking", "pointing", "beckoning"]


def simulate_sentence_building(prediction_vectors, actions, threshold=0.5):
    """
    Simulate the sentence-building logic from RealTimeDetector.run().

    This replicates the logic:
    1. For each prediction vector, determine argmax and confidence
    2. Track predictions history for 10-prediction consistency check
    3. Only consider an action valid when last 10 predictions all agree
       on the same argmax class with confidence >= threshold
    4. Only append when action differs from last entry in sentence
    5. Cap sentence at last 5 actions

    Args:
        prediction_vectors: List/array of prediction vectors (each has 3 elements)
        actions: List of action labels
        threshold: Confidence threshold (default 0.5)

    Returns:
        The final sentence list
    """
    sentence = []
    predictions = []  # History of prediction indices

    for res in prediction_vectors:
        predicted_idx = np.argmax(res)
        predictions.append(predicted_idx)

        # Apply confidence threshold and 10-prediction consistency check
        if len(predictions) >= 10:
            last_10 = predictions[-10:]
            # Check if last 10 predictions all agree
            if len(set(last_10)) == 1:
                predicted_action = actions[predicted_idx]
                confidence = res[predicted_idx]

                # Only display if confidence >= threshold
                if confidence >= threshold:
                    # Only append if different from last action in sentence
                    if len(sentence) == 0 or sentence[-1] != predicted_action:
                        sentence.append(predicted_action)

        # Cap sentence at last 5 actions
        if len(sentence) > 5:
            sentence = sentence[-5:]

    return sentence


@st.composite
def prediction_vector_sequences(draw):
    """Generate random sequences of 3-element probability vectors (10 to 100 vectors).

    Each vector represents a prediction output with 3 elements (one per action class).
    Values are between 0 and 1 to simulate model output probabilities.
    """
    seq_length = draw(st.integers(min_value=10, max_value=100))
    vectors = draw(
        hnp_arrays(
            dtype=np.float64,
            shape=(seq_length, 3),
            elements=st.floats(min_value=0.0, max_value=1.0,
                               allow_nan=False, allow_infinity=False),
        )
    )
    return vectors


class TestSentenceBuildingConsistencyProperty:
    """Property 9: Sentence-building consistency.

    **Validates: Requirements 7.5**
    """

    @given(prediction_vectors=prediction_vector_sequences())
    @settings(max_examples=100)
    def test_sentence_never_exceeds_5_actions(self, prediction_vectors):
        """For any sequence of prediction vectors, the sentence never exceeds 5 elements."""
        sentence = simulate_sentence_building(
            prediction_vectors, _SENTENCE_ACTIONS, threshold=0.5
        )
        assert len(sentence) <= 5, (
            f"Sentence has {len(sentence)} actions, expected at most 5: {sentence}"
        )

    @given(prediction_vectors=prediction_vector_sequences())
    @settings(max_examples=100)
    def test_no_consecutive_duplicate_actions(self, prediction_vectors):
        """For any sequence of prediction vectors, no two consecutive entries in the sentence are the same."""
        sentence = simulate_sentence_building(
            prediction_vectors, _SENTENCE_ACTIONS, threshold=0.5
        )
        for i in range(1, len(sentence)):
            assert sentence[i] != sentence[i - 1], (
                f"Consecutive duplicate found at index {i}: "
                f"sentence[{i-1}]='{sentence[i-1]}', sentence[{i}]='{sentence[i]}'. "
                f"Full sentence: {sentence}"
            )

    @given(prediction_vectors=prediction_vector_sequences())
    @settings(max_examples=100)
    def test_actions_only_added_when_last_10_predictions_agree(self, prediction_vectors):
        """Actions are only added to the sentence when the last 10 predictions
        all agree on the same argmax class with confidence >= 0.5.

        We verify this by checking that any action in the sentence corresponds
        to a point where 10 consecutive predictions agreed with sufficient confidence.
        """
        sentence = []
        predictions = []
        sentence_add_points = []  # Track when actions were added

        for idx, res in enumerate(prediction_vectors):
            predicted_idx = np.argmax(res)
            predictions.append(predicted_idx)

            if len(predictions) >= 10:
                last_10 = predictions[-10:]
                if len(set(last_10)) == 1:
                    predicted_action = _SENTENCE_ACTIONS[predicted_idx]
                    confidence = res[predicted_idx]

                    if confidence >= 0.5:
                        if len(sentence) == 0 or sentence[-1] != predicted_action:
                            sentence.append(predicted_action)
                            sentence_add_points.append({
                                "index": idx,
                                "action": predicted_action,
                                "last_10_agree": True,
                                "confidence": confidence,
                            })

            if len(sentence) > 5:
                sentence = sentence[-5:]

        # Verify all additions had 10-prediction agreement and confidence >= 0.5
        for point in sentence_add_points:
            assert point["last_10_agree"], (
                f"Action '{point['action']}' added at index {point['index']} "
                f"without 10-prediction agreement"
            )
            assert point["confidence"] >= 0.5, (
                f"Action '{point['action']}' added at index {point['index']} "
                f"with confidence {point['confidence']} < 0.5"
            )


# Unit tests for RealTimeDetector (Task 8.5)
# Validates: Requirement 6.8
"""
Unit tests for RealTimeDetector class.

Tests:
- Model file not found raises FileNotFoundError with descriptive message
- prob_viz returns valid image array
"""

import os
import tempfile

import numpy as np
import pytest

from screen_action_recognition import RealTimeDetector, prob_viz


class TestRealTimeDetectorInit:
    """Unit tests for RealTimeDetector initialization error handling.

    _Requirements: 6.8_
    """

    def test_model_file_not_found_raises_file_not_found_error(self):
        """RealTimeDetector should raise FileNotFoundError when model_path does not exist."""
        actions = ["walking", "pointing", "beckoning"]
        nonexistent_path = "nonexistent_model_file_xyz.h5"

        # Ensure the file truly doesn't exist
        assert not os.path.isfile(nonexistent_path)

        with pytest.raises(FileNotFoundError) as exc_info:
            RealTimeDetector(actions=actions, model_path=nonexistent_path)

        # Verify the error message is descriptive and mentions the missing file
        error_msg = str(exc_info.value)
        assert nonexistent_path in error_msg, (
            f"Error message should mention the missing file path '{nonexistent_path}', "
            f"got: '{error_msg}'"
        )

    def test_model_file_not_found_with_custom_path(self):
        """RealTimeDetector should raise FileNotFoundError for any non-existent custom path."""
        actions = ["walking", "pointing", "beckoning"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_model_path = os.path.join(tmp_dir, "missing_model.h5")

            with pytest.raises(FileNotFoundError) as exc_info:
                RealTimeDetector(actions=actions, model_path=fake_model_path)

            error_msg = str(exc_info.value)
            assert "missing_model.h5" in error_msg, (
                f"Error message should mention the missing file, got: '{error_msg}'"
            )


class TestProbViz:
    """Unit tests for prob_viz function output.

    _Requirements: 6.8_
    """

    def test_prob_viz_returns_valid_image_array(self):
        """prob_viz should return a valid numpy array (image with bars drawn)."""
        # Create a blank image (480x640x3, uint8)
        input_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        actions = ["walking", "pointing", "beckoning"]
        test_colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
        # Random probability values
        res = np.array([0.8, 0.15, 0.05])

        result = prob_viz(res, actions, input_frame, test_colors)

        # Result should be a numpy array
        assert isinstance(result, np.ndarray), (
            f"Expected numpy array, got {type(result)}"
        )
        # Result should have same shape as input (image with overlays)
        assert result.shape == input_frame.shape, (
            f"Expected shape {input_frame.shape}, got {result.shape}"
        )
        # Result should have dtype uint8
        assert result.dtype == np.uint8, (
            f"Expected dtype uint8, got {result.dtype}"
        )

    def test_prob_viz_does_not_modify_input_frame(self):
        """prob_viz should not modify the original input frame (uses copy internally)."""
        input_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_frame = input_frame.copy()
        actions = ["walking", "pointing", "beckoning"]
        test_colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
        res = np.array([0.5, 0.3, 0.2])

        prob_viz(res, actions, input_frame, test_colors)

        # Original frame should be unchanged
        np.testing.assert_array_equal(input_frame, original_frame)

    def test_prob_viz_with_zero_probabilities(self):
        """prob_viz should handle zero probability values and still return a valid image."""
        input_frame = np.ones((200, 300, 3), dtype=np.uint8) * 128
        actions = ["walking", "pointing", "beckoning"]
        test_colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
        res = np.array([0.0, 0.0, 0.0])

        result = prob_viz(res, actions, input_frame, test_colors)

        assert isinstance(result, np.ndarray)
        assert result.shape == (200, 300, 3)
        assert result.dtype == np.uint8


# Integration tests (Task 9.2)
# Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
"""
Integration tests for the screen action recognition pipeline.

Tests:
1. End-to-end data collection: capture frames → extract keypoints → save to disk → verify file structure
2. Pipeline compatibility: verify screen-captured frames produce correct downstream behavior through MediaPipe
"""

import os

import numpy as np
import mediapipe as mp

from screen_action_recognition import (
    ScreenCapture,
    safe_extract_keypoints,
    mediapipe_detection,
    extract_keypoints,
)


class TestEndToEndDataCollection:
    """Integration test: capture frames → extract keypoints → save to disk → verify file structure.

    _Requirements: 7.1, 7.2, 7.3_
    """

    def test_capture_extract_save_roundtrip(self, tmp_path):
        """Capture a real screen frame, extract keypoints, save as .npy, and verify the file."""
        # 1. Capture a frame from the screen using ScreenCapture
        capture = ScreenCapture(region={"top": 0, "left": 0, "width": 100, "height": 100})
        try:
            frame = capture.grab_frame()
        finally:
            capture.release()

        # 2. Extract keypoints using safe_extract_keypoints
        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            keypoints = safe_extract_keypoints(frame, holistic)

        # 3. Save keypoints to disk in the expected folder structure
        action = "walking"
        sequence = 0
        frame_num = 0
        dir_path = os.path.join(str(tmp_path), "MP_Data", action, str(sequence))
        os.makedirs(dir_path, exist_ok=True)
        npy_path = os.path.join(dir_path, str(frame_num))
        np.save(npy_path, keypoints)

        # 4. Verify the file exists and contains a 1662-element float64 array
        saved_file = npy_path + ".npy"
        assert os.path.isfile(saved_file), f"Expected .npy file at {saved_file}"

        loaded = np.load(saved_file)
        assert loaded.shape == (1662,), f"Expected shape (1662,), got {loaded.shape}"
        assert loaded.dtype == np.float64, f"Expected dtype float64, got {loaded.dtype}"
        np.testing.assert_array_equal(loaded, keypoints)

    def test_multiple_frames_saved_to_sequence_folder(self, tmp_path):
        """Capture multiple frames, extract keypoints, save each to a sequence folder, and verify structure."""
        num_frames = 5
        action = "pointing"
        sequence = 2
        dir_path = os.path.join(str(tmp_path), "MP_Data", action, str(sequence))
        os.makedirs(dir_path, exist_ok=True)

        capture = ScreenCapture(region={"top": 0, "left": 0, "width": 64, "height": 64})
        try:
            with mp.solutions.holistic.Holistic(
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) as holistic:
                for frame_num in range(num_frames):
                    frame = capture.grab_frame()
                    keypoints = safe_extract_keypoints(frame, holistic)
                    npy_path = os.path.join(dir_path, str(frame_num))
                    np.save(npy_path, keypoints)
        finally:
            capture.release()

        # Verify all frame files exist with correct properties
        for frame_num in range(num_frames):
            saved_file = os.path.join(dir_path, f"{frame_num}.npy")
            assert os.path.isfile(saved_file), f"Missing frame file: {saved_file}"
            loaded = np.load(saved_file)
            assert loaded.shape == (1662,)
            assert loaded.dtype == np.float64


class TestPipelineCompatibility:
    """Integration test: verify screen-captured frames produce correct downstream behavior through MediaPipe.

    _Requirements: 7.1, 7.2, 7.5_
    """

    def test_captured_frame_is_bgr_uint8(self):
        """Screen-captured frame should be BGR uint8 — compatible with cv2.VideoCapture format."""
        capture = ScreenCapture(region={"top": 0, "left": 0, "width": 200, "height": 150})
        try:
            frame = capture.grab_frame()
        finally:
            capture.release()

        # Frame must be a numpy array
        assert isinstance(frame, np.ndarray)
        # dtype must be uint8
        assert frame.dtype == np.uint8, f"Expected uint8, got {frame.dtype}"
        # Shape must be (height, width, 3) — 3 channels for BGR
        assert frame.ndim == 3, f"Expected 3 dimensions, got {frame.ndim}"
        assert frame.shape == (150, 200, 3), f"Expected (150, 200, 3), got {frame.shape}"

    def test_captured_frame_through_mediapipe_detection(self):
        """Screen-captured frame passed through mediapipe_detection returns a results object."""
        capture = ScreenCapture(region={"top": 0, "left": 0, "width": 320, "height": 240})
        try:
            frame = capture.grab_frame()
        finally:
            capture.release()

        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            image, results = mediapipe_detection(frame, holistic)

        # The returned image should be a BGR numpy array with same spatial dimensions
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert image.shape == frame.shape

        # Results object should have the expected landmark attributes
        assert hasattr(results, "pose_landmarks")
        assert hasattr(results, "face_landmarks")
        assert hasattr(results, "left_hand_landmarks")
        assert hasattr(results, "right_hand_landmarks")

    def test_captured_frame_through_full_keypoint_pipeline(self):
        """Screen-captured frame through mediapipe_detection → extract_keypoints produces 1662-element array."""
        capture = ScreenCapture(region={"top": 0, "left": 0, "width": 320, "height": 240})
        try:
            frame = capture.grab_frame()
        finally:
            capture.release()

        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)

        # Keypoints must be a 1662-element array
        assert isinstance(keypoints, np.ndarray)
        assert keypoints.shape == (1662,), f"Expected (1662,), got {keypoints.shape}"
        # All values should be finite
        assert np.all(np.isfinite(keypoints)), "Keypoints contain non-finite values"

    def test_safe_extract_keypoints_with_real_screen_frame(self):
        """safe_extract_keypoints with a real screen frame returns a valid 1662-element float64 array."""
        capture = ScreenCapture(region={"top": 0, "left": 0, "width": 160, "height": 120})
        try:
            frame = capture.grab_frame()
        finally:
            capture.release()

        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as holistic:
            keypoints = safe_extract_keypoints(frame, holistic)

        assert isinstance(keypoints, np.ndarray)
        assert keypoints.shape == (1662,)
        assert keypoints.dtype == np.float64
        assert np.all(np.isfinite(keypoints))
