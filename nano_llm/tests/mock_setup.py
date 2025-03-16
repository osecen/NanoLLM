import sys
from unittest.mock import MagicMock

# Mock the torch module
sys.modules['torch'] = MagicMock()
sys.modules['torch.cuda'] = MagicMock() 