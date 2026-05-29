"""
Unit tests for ScreenCapture class.

Tests cover:
- Full-monitor capture when no region specified (Requirement 1.4)
- mss import failure raises ImportError with correct message (Requirement 8.3)
- 30 consecutive failures raises RuntimeError (Requirements 1.5, 1.6)
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock


class TestScreenCaptureFullMonitor:
    """Test full-monitor capture when no region is specified (Requirement 1.4)."""

    def test_defaults_to_full_primary_monitor(self):
        """When no region is specified, ScreenCapture should use the full primary monitor."""
        mock_sct_instance = MagicMock()
        mock_sct_instance.monitors = [
            {"top": 0, "left": 0, "width": 3840, "height": 2160},  # all monitors
            {"top": 0, "left": 0, "width": 1920, "height": 1080},  # primary monitor
        ]

        with patch("screen_action_recognition.mss") as mock_mss_module:
            # Patch the import inside __init__
            mock_mss_lib = MagicMock()
            mock_mss_lib.mss.return_value = mock_sct_instance

            with patch.dict("sys.modules", {"mss": mock_mss_lib}):
                with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: mock_mss_lib if name == "mss" else __builtins__.__import__(name, *args, **kwargs)):
                    # We need a cleaner approach - patch at the class level
                    pass

        # Use a direct approach: patch mss.mss() call inside the class
        import screen_action_recognition

        with patch.object(screen_action_recognition, "mss") as _:
            # Patch the dynamic import inside __init__
            mock_mss_mod = MagicMock()
            mock_mss_mod.mss.return_value = mock_sct_instance

            with patch("builtins.__import__", wraps=__import__) as mock_import:
                def side_effect(name, *args, **kwargs):
                    if name == "mss":
                        return mock_mss_mod
                    return __import__(name, *args, **kwargs)

                mock_import.side_effect = side_effect

                capture = screen_action_recognition.ScreenCapture(region=None)

                # Verify the monitor is set to full primary monitor dimensions
                assert capture._monitor["top"] == 0
                assert capture._monitor["left"] == 0
                assert capture._monitor["width"] == 1920
                assert capture._monitor["height"] == 1080

    def test_grab_frame_returns_bgr_array(self):
        """grab_frame should return a BGR numpy array with correct shape."""
        mock_sct_instance = MagicMock()
        mock_sct_instance.monitors = [
            {"top": 0, "left": 0, "width": 3840, "height": 2160},
            {"top": 0, "left": 0, "width": 1920, "height": 1080},
        ]

        # Create a fake BGRA screenshot (100x100 pixels)
        fake_bgra = np.zeros((100, 100, 4), dtype=np.uint8)
        fake_bgra[:, :, 0] = 255  # B channel
        fake_bgra[:, :, 1] = 128  # G channel
        fake_bgra[:, :, 2] = 64   # R channel
        fake_bgra[:, :, 3] = 255  # A channel

        mock_screenshot = MagicMock()
        # Make np.array(screenshot) return our fake BGRA data
        mock_screenshot.__array__ = lambda self, dtype=None: fake_bgra

        mock_sct_instance.grab.return_value = mock_screenshot

        import screen_action_recognition

        mock_mss_mod = MagicMock()
        mock_mss_mod.mss.return_value = mock_sct_instance

        with patch("builtins.__import__", wraps=__import__) as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == "mss":
                    return mock_mss_mod
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            capture = screen_action_recognition.ScreenCapture(region=None)

        # Now mock np.array to handle the screenshot object
        with patch("numpy.array", return_value=fake_bgra):
            frame = capture.grab_frame()

        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (100, 100, 3)  # BGR, no alpha


class TestScreenCaptureMssImportFailure:
    """Test mss import failure raises ImportError with correct message (Requirement 8.3)."""

    def test_raises_import_error_when_mss_not_installed(self):
        """If mss cannot be imported, ScreenCapture should raise ImportError with install instructions."""
        import screen_action_recognition

        with patch("builtins.__import__", wraps=__import__) as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == "mss":
                    raise ImportError("No module named 'mss'")
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            with pytest.raises(ImportError) as exc_info:
                screen_action_recognition.ScreenCapture(region=None)

            assert "mss is required for screen capture" in str(exc_info.value)
            assert "pip install mss" in str(exc_info.value)


class TestScreenCaptureConsecutiveFailures:
    """Test 30 consecutive failures raises RuntimeError (Requirements 1.5, 1.6)."""

    def test_raises_runtime_error_after_30_consecutive_failures(self):
        """After 30 consecutive grab failures, RuntimeError should be raised."""
        mock_sct_instance = MagicMock()
        mock_sct_instance.monitors = [
            {"top": 0, "left": 0, "width": 3840, "height": 2160},
            {"top": 0, "left": 0, "width": 1920, "height": 1080},
        ]
        # Make grab always raise an exception
        mock_sct_instance.grab.side_effect = Exception("Capture failed")

        import screen_action_recognition

        mock_mss_mod = MagicMock()
        mock_mss_mod.mss.return_value = mock_sct_instance

        with patch("builtins.__import__", wraps=__import__) as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == "mss":
                    return mock_mss_mod
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            capture = screen_action_recognition.ScreenCapture(region=None)

        # Call grab_frame 29 times - should raise the individual exception but not RuntimeError
        for i in range(29):
            with pytest.raises(Exception, match="Capture failed"):
                capture.grab_frame()

        # The 30th failure should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            capture.grab_frame()

        assert "30 consecutive" in str(exc_info.value)
        assert "unavailable" in str(exc_info.value)

    def test_successful_grab_resets_failure_counter(self):
        """A successful grab should reset the consecutive failure counter."""
        mock_sct_instance = MagicMock()
        mock_sct_instance.monitors = [
            {"top": 0, "left": 0, "width": 3840, "height": 2160},
            {"top": 0, "left": 0, "width": 1920, "height": 1080},
        ]

        # Create a fake BGRA screenshot
        fake_bgra = np.zeros((1080, 1920, 4), dtype=np.uint8)
        mock_screenshot = MagicMock()
        mock_screenshot.__array__ = lambda self, dtype=None: fake_bgra

        # First 29 calls fail, then one succeeds, then 29 more fail
        call_count = [0]

        def grab_side_effect(monitor):
            call_count[0] += 1
            if call_count[0] <= 29:
                raise Exception("Capture failed")
            elif call_count[0] == 30:
                return mock_screenshot
            else:
                raise Exception("Capture failed")

        mock_sct_instance.grab.side_effect = grab_side_effect

        import screen_action_recognition

        mock_mss_mod = MagicMock()
        mock_mss_mod.mss.return_value = mock_sct_instance

        with patch("builtins.__import__", wraps=__import__) as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == "mss":
                    return mock_mss_mod
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = side_effect

            capture = screen_action_recognition.ScreenCapture(region=None)

        # 29 failures
        for i in range(29):
            with pytest.raises(Exception, match="Capture failed"):
                capture.grab_frame()

        # 1 success - should reset counter
        with patch("numpy.array", return_value=fake_bgra):
            frame = capture.grab_frame()
        assert frame is not None

        # Now 29 more failures should NOT raise RuntimeError
        for i in range(29):
            with pytest.raises(Exception, match="Capture failed"):
                capture.grab_frame()

        # But the 30th after reset SHOULD raise RuntimeError
        with pytest.raises(RuntimeError):
            capture.grab_frame()
