# Mock torch before any imports
import sys
from unittest.mock import MagicMock

# Create mock torch module
mock_torch = MagicMock()
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available = MagicMock(return_value=True)
mock_torch.cuda.Stream = MagicMock()
mock_torch.cuda.StreamPriority = MagicMock()
mock_torch.cuda.StreamPriority.HIGH = 0
mock_torch.cuda.StreamPriority.NORMAL = -1
mock_torch.cuda.StreamPriority.LOW = -2

# Add to sys.modules
sys.modules['torch'] = mock_torch
sys.modules['torch.cuda'] = mock_torch.cuda

# Now import unittest and other modules
import unittest
from unittest.mock import patch
import logging
import os
import ast
import re
import glob

# Add the NanoLLM directory to the Python path
NANO_LLM_PATH = "/root/repo/NanoLLM"
if NANO_LLM_PATH not in sys.path:
    sys.path.insert(0, NANO_LLM_PATH)

# If the test is being run from the held_out_tests directory, adjust paths accordingly
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(TEST_DIR) == "held_out_tests":
    print(f"Running from held_out_tests directory: {TEST_DIR}")
else:
    print(f"Running from directory: {TEST_DIR}")

# Create a minimal mock Plugin class for testing
class MockPlugin:
    def __init__(self, name=None, resource_priority=0, **kwargs):
        self.name = name or "MockPlugin"
        self.resource_priority = resource_priority
        self.cuda_stream = None
        
        # Simulate the CUDA stream creation logic
        if mock_torch.cuda.is_available() and resource_priority > 0:
            # First create a basic stream as fallback
            self.cuda_stream = mock_torch.cuda.Stream()
            
            # Then try to create a prioritized stream
            if resource_priority >= 9:
                priority = mock_torch.cuda.StreamPriority.HIGH
            elif resource_priority >= 5:
                priority = mock_torch.cuda.StreamPriority.NORMAL
            else:
                priority = mock_torch.cuda.StreamPriority.LOW
                
            self.cuda_stream = mock_torch.cuda.Stream(priority=priority)

class PluginResourcePriorityTestBase(unittest.TestCase):
    """Base class for testing plugin resource priority."""
    
    def _find_file(self, filename):
        """Find a file in the NanoLLM directory structure."""
        # Try to find the file using glob
        pattern = os.path.join(NANO_LLM_PATH, "**", filename)
        matches = glob.glob(pattern, recursive=True)
        
        if matches:
            return matches[0]
        return None
    
    def _check_resource_priority_in_file(self, filepath):
        """Check if a file contains a class that passes resource_priority to super().__init__."""
        print(f"Checking file: {filepath}")
        
        # Read the file content
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Find super().__init__ calls with resource_priority
        init_pattern = r"super\(\)\.__init__\([^)]*resource_priority\s*=\s*[^,)]*[^)]*\)"
        match = re.search(init_pattern, content)
        
        if match:
            print(f"Found super().__init__ call with resource_priority: {match.group(0)}")
            # If we found a super().__init__ call with resource_priority, that's good enough
            # The class might inherit from Plugin indirectly
            return True
        else:
            print("No super().__init__ call with resource_priority found")
            return False
    
    def _test_plugin_resource_priority(self, filename, plugin_name):
        """Generic method to test a plugin's resource priority."""
        # Find the plugin file
        plugin_file = self._find_file(filename)
        self.assertIsNotNone(plugin_file, f"{filename} file not found")
        
        # Check if resource_priority is passed to super().__init__
        result = self._check_resource_priority_in_file(plugin_file)
        self.assertTrue(result, f"{plugin_name} plugin does not correctly pass resource_priority to super().__init__")


class TestAudioPlugins(PluginResourcePriorityTestBase):
    """Tests for plugins in the audio category."""
    
    def test_audio_input_resource_priority(self):
        """Test that the AudioInput plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("audio_input.py", "AudioInput")
    
    def test_audio_output_resource_priority(self):
        """Test that the AudioOutput plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("audio_output.py", "AudioOutput")
    
    def test_web_audio_resource_priority(self):
        """Test that the WebAudio plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("web_audio.py", "WebAudio")


class TestDataPlugins(PluginResourcePriorityTestBase):
    """Tests for plugins in the data category."""
    
    def test_data_logger_resource_priority(self):
        """Test that the DataLogger plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("data_logger.py", "DataLogger")
    
    def test_data_table_resource_priority(self):
        """Test that the DataTable plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("data_table.py", "DataTable")
    
    def test_deduplicate_resource_priority(self):
        """Test that the Deduplicate plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("deduplicate.py", "Deduplicate")
    
    def test_event_filter_resource_priority(self):
        """Test that the EventFilter plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("event_filter.py", "EventFilter")
    
    def test_mux_resource_priority(self):
        """Test that the Mux plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("mux.py", "Mux")
    
    def test_nanodb_resource_priority(self):
        """Test that the NanoDB plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("nanodb.py", "NanoDB")


class TestLLMPlugins(PluginResourcePriorityTestBase):
    """Tests for plugins in the LLM category."""
    
    def test_auto_prompt_resource_priority(self):
        """Test that the AutoPrompt plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("auto_prompt.py", "AutoPrompt")
    
    def test_nano_llm_resource_priority(self):
        """Test that the NanoLLM plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("nano_llm.py", "NanoLLM")
    
    def test_nano_vla_resource_priority(self):
        """Test that the NanoVLA plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("nano_vla.py", "NanoVLA")
    
    def test_text_stream_resource_priority(self):
        """Test that the TextStream plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("text_stream.py", "TextStream")
    
    def test_user_prompt_resource_priority(self):
        """Test that the UserPrompt plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("user_prompt.py", "UserPrompt")


class TestRoboticsPlugins(PluginResourcePriorityTestBase):
    """Tests for plugins in the robotics category."""
    
    def test_mimicgen_resource_priority(self):
        """Test that the MimicGen plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("mimicgen.py", "MimicGen")
    
    def test_robot_dataset_resource_priority(self):
        """Test that the RobotDataset plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("robot_dataset.py", "RobotDataset")
    
    def test_ros_connector_resource_priority(self):
        """Test that the ROSConnector plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("ros_connector.py", "ROSConnector")


class TestSpeechPlugins(PluginResourcePriorityTestBase):
    """Tests for plugins in the speech category."""
    
    def test_auto_asr_resource_priority(self):
        """Test that the AutoASR plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("auto_asr.py", "AutoASR")
    
    def test_auto_tts_resource_priority(self):
        """Test that the AutoTTS plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("auto_tts.py", "AutoTTS")
    
    def test_fastpitch_tts_resource_priority(self):
        """Test that the FastPitchTTS plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("fastpitch_tts.py", "FastPitchTTS")
    
    def test_piper_tts_resource_priority(self):
        """Test that the PiperTTS plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("piper_tts.py", "PiperTTS")
    
    def test_riva_asr_resource_priority(self):
        """Test that the RivaASR plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("riva_asr.py", "RivaASR")
    
    def test_riva_tts_resource_priority(self):
        """Test that the RivaTTS plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("riva_tts.py", "RivaTTS")
    
    def test_vad_filter_resource_priority(self):
        """Test that the VADFilter plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("vad_filter.py", "VADFilter")
    
    def test_whisper_asr_resource_priority(self):
        """Test that the WhisperASR plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("whisper_asr.py", "WhisperASR")
    
    def test_xtts_resource_priority(self):
        """Test that the XTTS plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("xtts.py", "XTTS")


class TestVideoPlugins(PluginResourcePriorityTestBase):
    """Tests for plugins in the video category."""
    
    def test_rate_limit_resource_priority(self):
        """Test that the RateLimit plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("rate_limit.py", "RateLimit")
    
    def test_text_overlay_resource_priority(self):
        """Test that the TextOverlay plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("text_overlay.py", "TextOverlay")
    
    def test_video_output_resource_priority(self):
        """Test that the VideoOutput plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("video_output.py", "VideoOutput")
    
    def test_video_source_resource_priority(self):
        """Test that the VideoSource plugin passes resource_priority to super().__init__."""
        self._test_plugin_resource_priority("video_source.py", "VideoSource")


if __name__ == '__main__':
    unittest.main()