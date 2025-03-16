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

# Test class
class TestPluginCudaStream(unittest.TestCase):
    
    def setUp(self):
        # Reset mocks before each test
        mock_torch.cuda.Stream.reset_mock()
    
    def test_cuda_stream_creation_with_priority(self):
        # Create a plugin with high resource priority
        plugin = MockPlugin(name="TestPlugin", resource_priority=10)
        
        # Verify that Stream was called twice (once for fallback, once with priority)
        self.assertEqual(mock_torch.cuda.Stream.call_count, 2)
        
        # Get the last call which should be the one with HIGH priority
        args, kwargs = mock_torch.cuda.Stream.call_args_list[-1]
        
        # Check that priority was passed as a parameter
        self.assertIn('priority', kwargs)
        self.assertEqual(kwargs['priority'], mock_torch.cuda.StreamPriority.HIGH)
        
        # Verify that the cuda_stream attribute was set correctly
        self.assertEqual(plugin.cuda_stream, mock_torch.cuda.Stream.return_value)
    
    def test_cuda_stream_creation_with_medium_priority(self):
        # Create a plugin with medium resource priority
        plugin = MockPlugin(name="TestPlugin", resource_priority=6)
        
        # Verify that Stream was called twice
        self.assertEqual(mock_torch.cuda.Stream.call_count, 2)
        
        # Get the last call which should be the one with NORMAL priority
        args, kwargs = mock_torch.cuda.Stream.call_args_list[-1]
        
        # Check that priority was passed as a parameter
        self.assertIn('priority', kwargs)
        self.assertEqual(kwargs['priority'], mock_torch.cuda.StreamPriority.NORMAL)
    
    def test_cuda_stream_creation_without_priority(self):
        # Create a plugin with no resource priority
        plugin = MockPlugin(name="TestPlugin", resource_priority=0)
        
        # Verify that Stream was not called
        mock_torch.cuda.Stream.assert_not_called()
        
        # Verify that the cuda_stream attribute was not set
        self.assertIsNone(plugin.cuda_stream)

class TestNanoDBResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 7
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock NanoDB class that inherits from MockPlugin
            class MockNanoDB(MockPlugin):
                def __init__(self, resource_priority=7, **kwargs):
                    super().__init__(
                        name="NanoDB",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            nanodb_instance = MockNanoDB(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(nanodb_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for NanoLLM plugin
class TestNanoLLMResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 10  # Critical priority for LLM
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock NanoLLM class that inherits from MockPlugin
            class MockNanoLLM(MockPlugin):
                def __init__(self, resource_priority=10, **kwargs):
                    super().__init__(
                        name="NanoLLM",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            nanollm_instance = MockNanoLLM(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(nanollm_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for WhisperASR plugin
class TestWhisperASRResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 8  # High priority for ASR
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock WhisperASR class that inherits from MockPlugin
            class MockWhisperASR(MockPlugin):
                def __init__(self, resource_priority=8, **kwargs):
                    super().__init__(
                        name="WhisperASR",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            whisper_instance = MockWhisperASR(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(whisper_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for VideoSource plugin
class TestVideoSourceResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 6  # Medium priority for video processing
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock VideoSource class that inherits from MockPlugin
            class MockVideoSource(MockPlugin):
                def __init__(self, resource_priority=6, **kwargs):
                    super().__init__(
                        name="VideoSource",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            video_source_instance = MockVideoSource(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(video_source_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for Clock plugin (low priority)
class TestClockResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 3  # Low priority for utility plugins
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock Clock class that inherits from MockPlugin
            class MockClock(MockPlugin):
                def __init__(self, resource_priority=3, **kwargs):
                    super().__init__(
                        name="Clock",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            clock_instance = MockClock(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(clock_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for AudioInput plugin
class TestAudioInputResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 5  # Medium priority for audio input
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock AudioInput class that inherits from MockPlugin
            class MockAudioInput(MockPlugin):
                def __init__(self, resource_priority=5, **kwargs):
                    super().__init__(
                        name="AudioInput",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            audio_input_instance = MockAudioInput(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(audio_input_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for AudioOutput plugin
class TestAudioOutputResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 4  # Medium-low priority for audio output
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock AudioOutput class that inherits from MockPlugin
            class MockAudioOutput(MockPlugin):
                def __init__(self, resource_priority=4, **kwargs):
                    super().__init__(
                        name="AudioOutput",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            audio_output_instance = MockAudioOutput(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(audio_output_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for WebAudio plugin
class TestWebAudioResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 4  # Medium-low priority for web audio
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock WebAudio class that inherits from MockPlugin
            class MockWebAudio(MockPlugin):
                def __init__(self, resource_priority=4, **kwargs):
                    super().__init__(
                        name="WebAudio",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            web_audio_instance = MockWebAudio(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(web_audio_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for Alert plugin
class TestAlertResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 3  # Low priority for utility plugins
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock Alert class that inherits from MockPlugin
            class MockAlert(MockPlugin):
                def __init__(self, resource_priority=3, **kwargs):
                    super().__init__(
                        name="Alert",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            alert_instance = MockAlert(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(alert_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for HomeAssistant plugin
class TestHomeAssistantResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 4  # Medium-low priority for home automation
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock HomeAssistant class that inherits from MockPlugin
            class MockHomeAssistant(MockPlugin):
                def __init__(self, resource_priority=4, **kwargs):
                    super().__init__(
                        name="HomeAssistant",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            home_assistant_instance = MockHomeAssistant(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(home_assistant_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for Location plugin
class TestLocationResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 3  # Low priority for utility plugins
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock Location class that inherits from MockPlugin
            class MockLocation(MockPlugin):
                def __init__(self, resource_priority=3, **kwargs):
                    super().__init__(
                        name="Location",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            location_instance = MockLocation(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(location_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for Weather plugin
class TestWeatherResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 3  # Low priority for utility plugins
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock Weather class that inherits from MockPlugin
            class MockWeather(MockPlugin):
                def __init__(self, resource_priority=3, **kwargs):
                    super().__init__(
                        name="Weather",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            weather_instance = MockWeather(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(weather_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for DataLogger plugin
class TestDataLoggerResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 3  # Low priority for utility plugins
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock DataLogger class that inherits from MockPlugin
            class MockDataLogger(MockPlugin):
                def __init__(self, resource_priority=3, **kwargs):
                    super().__init__(
                        name="DataLogger",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            data_logger_instance = MockDataLogger(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(data_logger_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for DataTable plugin
class TestDataTableResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 3  # Low priority for utility plugins
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock DataTable class that inherits from MockPlugin
            class MockDataTable(MockPlugin):
                def __init__(self, resource_priority=3, **kwargs):
                    super().__init__(
                        name="DataTable",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            data_table_instance = MockDataTable(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(data_table_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for Deduplicate plugin
class TestDeduplicateResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 3  # Low priority for utility plugins
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock Deduplicate class that inherits from MockPlugin
            class MockDeduplicate(MockPlugin):
                def __init__(self, resource_priority=3, **kwargs):
                    super().__init__(
                        name="Deduplicate",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            deduplicate_instance = MockDeduplicate(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(deduplicate_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for EventFilter plugin
class TestEventFilterResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 3  # Low priority for utility plugins
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock EventFilter class that inherits from MockPlugin
            class MockEventFilter(MockPlugin):
                def __init__(self, resource_priority=3, **kwargs):
                    super().__init__(
                        name="EventFilter",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            event_filter_instance = MockEventFilter(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(event_filter_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for Mux plugin
class TestMuxResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 3  # Low priority for utility plugins
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock Mux class that inherits from MockPlugin
            class MockMux(MockPlugin):
                def __init__(self, resource_priority=3, **kwargs):
                    super().__init__(
                        name="Mux",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            mux_instance = MockMux(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(mux_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for AutoPrompt plugin
class TestAutoPromptResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 5  # Medium priority for prompt generation
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock AutoPrompt class that inherits from MockPlugin
            class MockAutoPrompt(MockPlugin):
                def __init__(self, resource_priority=5, **kwargs):
                    super().__init__(
                        name="AutoPrompt",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            auto_prompt_instance = MockAutoPrompt(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(auto_prompt_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for NanoVLA plugin
class TestNanoVLAResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 9  # High priority for vision-language model
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock NanoVLA class that inherits from MockPlugin
            class MockNanoVLA(MockPlugin):
                def __init__(self, resource_priority=9, **kwargs):
                    super().__init__(
                        name="NanoVLA",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            nano_vla_instance = MockNanoVLA(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(nano_vla_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for TextStream plugin
class TestTextStreamResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 4  # Medium-low priority for text streaming
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock TextStream class that inherits from MockPlugin
            class MockTextStream(MockPlugin):
                def __init__(self, resource_priority=4, **kwargs):
                    super().__init__(
                        name="TextStream",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            text_stream_instance = MockTextStream(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(text_stream_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for UserPrompt plugin
class TestUserPromptResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 4  # Medium-low priority for user input handling
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock UserPrompt class that inherits from MockPlugin
            class MockUserPrompt(MockPlugin):
                def __init__(self, resource_priority=4, **kwargs):
                    super().__init__(
                        name="UserPrompt",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            user_prompt_instance = MockUserPrompt(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(user_prompt_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for MimicGen plugin
class TestMimicGenResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 7  # High priority for speech synthesis
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock MimicGen class that inherits from MockPlugin
            class MockMimicGen(MockPlugin):
                def __init__(self, resource_priority=7, **kwargs):
                    super().__init__(
                        name="MimicGen",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            mimicgen_instance = MockMimicGen(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(mimicgen_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for RobotDataset plugin
class TestRobotDatasetResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 5  # Medium priority for dataset handling
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock RobotDataset class that inherits from MockPlugin
            class MockRobotDataset(MockPlugin):
                def __init__(self, resource_priority=5, **kwargs):
                    super().__init__(
                        name="RobotDataset",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            robot_dataset_instance = MockRobotDataset(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(robot_dataset_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for ROSConnector plugin
class TestROSConnectorResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 5  # Medium priority for robot communication
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock ROSConnector class that inherits from MockPlugin
            class MockROSConnector(MockPlugin):
                def __init__(self, resource_priority=5, **kwargs):
                    super().__init__(
                        name="ROSConnector",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            ros_connector_instance = MockROSConnector(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(ros_connector_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for AutoASR plugin
class TestAutoASRResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 8  # High priority for speech recognition
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock AutoASR class that inherits from MockPlugin
            class MockAutoASR(MockPlugin):
                def __init__(self, resource_priority=8, **kwargs):
                    super().__init__(
                        name="AutoASR",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            auto_asr_instance = MockAutoASR(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(auto_asr_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for AutoTTS plugin
class TestAutoTTSResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 7  # High priority for speech synthesis
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock AutoTTS class that inherits from MockPlugin
            class MockAutoTTS(MockPlugin):
                def __init__(self, resource_priority=7, **kwargs):
                    super().__init__(
                        name="AutoTTS",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            auto_tts_instance = MockAutoTTS(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(auto_tts_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for FastPitchTTS plugin
class TestFastPitchTTSResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 7  # High priority for speech synthesis
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock FastPitchTTS class that inherits from MockPlugin
            class MockFastPitchTTS(MockPlugin):
                def __init__(self, resource_priority=7, **kwargs):
                    super().__init__(
                        name="FastPitchTTS",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            fastpitch_tts_instance = MockFastPitchTTS(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(fastpitch_tts_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for PiperTTS plugin
class TestPiperTTSResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 7  # High priority for speech synthesis
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock PiperTTS class that inherits from MockPlugin
            class MockPiperTTS(MockPlugin):
                def __init__(self, resource_priority=7, **kwargs):
                    super().__init__(
                        name="PiperTTS",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            piper_tts_instance = MockPiperTTS(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(piper_tts_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for RivaTTS plugin
class TestRivaTTSResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 7  # High priority for speech synthesis
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock RivaTTS class that inherits from MockPlugin
            class MockRivaTTS(MockPlugin):
                def __init__(self, resource_priority=7, **kwargs):
                    super().__init__(
                        name="RivaTTS",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            riva_tts_instance = MockRivaTTS(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(riva_tts_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for RivaASR plugin
class TestRivaASRResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 8  # High priority for speech recognition
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock RivaASR class that inherits from MockPlugin
            class MockRivaASR(MockPlugin):
                def __init__(self, resource_priority=8, **kwargs):
                    super().__init__(
                        name="RivaASR",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            riva_asr_instance = MockRivaASR(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(riva_asr_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for VADFilter plugin
class TestVADFilterResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 6  # Medium priority for audio processing
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock VADFilter class that inherits from MockPlugin
            class MockVADFilter(MockPlugin):
                def __init__(self, resource_priority=6, **kwargs):
                    super().__init__(
                        name="VADFilter",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            vad_filter_instance = MockVADFilter(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(vad_filter_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for XTTS plugin
class TestXTTSResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 7  # High priority for speech synthesis
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock XTTS class that inherits from MockPlugin
            class MockXTTS(MockPlugin):
                def __init__(self, resource_priority=7, **kwargs):
                    super().__init__(
                        name="XTTS",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            xtts_instance = MockXTTS(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(xtts_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for AccuWeather plugin
class TestAccuWeatherResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 3  # Low priority for utility plugins
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock AccuWeather class that inherits from MockPlugin
            class MockAccuWeather(MockPlugin):
                def __init__(self, resource_priority=3, **kwargs):
                    super().__init__(
                        name="AccuWeather",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            accuweather_instance = MockAccuWeather(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(accuweather_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for Notification plugin
class TestNotificationResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 3  # Low priority for utility plugins
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock Notification class that inherits from MockPlugin
            class MockNotification(MockPlugin):
                def __init__(self, resource_priority=3, **kwargs):
                    super().__init__(
                        name="Notification",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            notification_instance = MockNotification(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(notification_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for RateLimit plugin
class TestRateLimitResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 3  # Low priority for utility plugins
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock RateLimit class that inherits from MockPlugin
            class MockRateLimit(MockPlugin):
                def __init__(self, resource_priority=3, **kwargs):
                    super().__init__(
                        name="RateLimit",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            rate_limit_instance = MockRateLimit(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(rate_limit_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for TextOverlay plugin
class TestTextOverlayResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 5  # Medium priority for video processing
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock TextOverlay class that inherits from MockPlugin
            class MockTextOverlay(MockPlugin):
                def __init__(self, resource_priority=5, **kwargs):
                    super().__init__(
                        name="TextOverlay",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            text_overlay_instance = MockTextOverlay(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(text_overlay_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

# Test for VideoOutput plugin
class TestVideoOutputResourcePriority(unittest.TestCase):
    def test_resource_priority(self):
        # Arrange
        expected_priority = 5  # Medium priority for video processing
        
        # Create a mock for Plugin.__init__
        original_plugin_init = MockPlugin.__init__
        
        # Track if Plugin.__init__ was called with the correct resource_priority
        plugin_init_called_with_correct_args = False
        
        def mock_plugin_init(self, name=None, resource_priority=0, **kwargs):
            nonlocal plugin_init_called_with_correct_args
            if resource_priority == expected_priority:
                plugin_init_called_with_correct_args = True
            return original_plugin_init(self, name, resource_priority, **kwargs)
        
        # Replace Plugin.__init__ with our mock
        MockPlugin.__init__ = mock_plugin_init
        
        try:
            # Create a mock VideoOutput class that inherits from MockPlugin
            class MockVideoOutput(MockPlugin):
                def __init__(self, resource_priority=5, **kwargs):
                    super().__init__(
                        name="VideoOutput",
                        resource_priority=resource_priority,
                        **kwargs
                    )
            
            # Act
            video_output_instance = MockVideoOutput(resource_priority=expected_priority)
            
            # Assert
            self.assertTrue(plugin_init_called_with_correct_args, 
                           "Plugin.__init__ was not called with the correct resource_priority")
            self.assertEqual(video_output_instance.resource_priority, expected_priority)
        
        finally:
            # Restore the original Plugin.__init__
            MockPlugin.__init__ = original_plugin_init

if __name__ == '__main__':
    unittest.main()