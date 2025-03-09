#!/usr/bin/env python3
import os
import gc
import time
import threading
import logging
import weakref
import ctypes

import torch
import numpy as np

class PrioritizedCudaStream:
    """
    A wrapper around torch.cuda.Stream that creates a stream with priority.
    
    In CUDA, stream priorities need to be set at creation time, not after a stream is
    already created. This class handles creating streams with different priorities based
    on the PyTorch version and available CUDA features.
    """
    # Define priority constants (higher number = higher priority)
    # These map to CUDA's internal values but might differ across PyTorch versions
    PRIORITY_HIGH = 0
    PRIORITY_NORMAL = -1
    PRIORITY_LOW = -2
    
    # Track whether we've already checked for priority support
    _priority_support_checked = False
    _priority_support_available = False
    _priority_needs_kwargs = False
    
    @classmethod
    def _check_priority_support(cls):
        """Check if this PyTorch version supports stream priorities at creation time"""
        if cls._priority_support_checked:
            return cls._priority_support_available
            
        # Only do this check once
        cls._priority_support_checked = True
        
        # Check for StreamPriority enum
        if hasattr(torch.cuda, 'StreamPriority'):
            logging.info("CUDA stream priorities supported through StreamPriority enum")
            
            # Update our constants with PyTorch's values if available
            if hasattr(torch.cuda.StreamPriority, 'HIGH'):
                cls.PRIORITY_HIGH = torch.cuda.StreamPriority.HIGH
            if hasattr(torch.cuda.StreamPriority, 'NORMAL'):
                cls.PRIORITY_NORMAL = torch.cuda.StreamPriority.NORMAL
            if hasattr(torch.cuda.StreamPriority, 'LOW'):
                cls.PRIORITY_LOW = torch.cuda.StreamPriority.LOW
                
            # Check if Stream constructor accepts priority
            import inspect
            stream_params = inspect.signature(torch.cuda.Stream).parameters
            if 'priority' in stream_params:
                logging.info("CUDA stream priorities supported directly in Stream constructor")
                cls._priority_support_available = True
                return True
                
            # Test if Stream accepts priority as a kwarg even if not in signature
            try:
                test_stream = torch.cuda.Stream(priority=cls.PRIORITY_NORMAL)
                logging.info("CUDA stream priorities supported through kwargs in Stream constructor")
                cls._priority_needs_kwargs = True
                cls._priority_support_available = True
                return True
            except Exception:
                logging.warning("CUDA stream priorities not supported in Stream constructor despite StreamPriority enum")
                
        else:
            logging.warning("StreamPriority enum not available in this PyTorch version")
            
        # Check if we can use the CUDA runtime API directly
        try:
            # Create a test stream
            test_stream = torch.cuda.Stream()
            
            # Find CUDA runtime library
            cuda_rt = None
            cuda_lib_paths = [
                'libcudart.so', 
                'libcudart.so.12',
                'libcudart.so.11',
                '/usr/local/cuda/lib64/libcudart.so',
                '/usr/local/cuda/lib64/libcudart.so.12',
                '/usr/local/cuda/lib64/libcudart.so.11'
            ]
            
            for lib_path in cuda_lib_paths:
                try:
                    cuda_rt = ctypes.CDLL(lib_path)
                    if hasattr(cuda_rt, 'cudaStreamCreateWithPriority'):
                        logging.info(f"Found CUDA runtime with cudaStreamCreateWithPriority at {lib_path}")
                        cls._priority_support_available = True
                        return True
                except Exception:
                    continue
        except Exception as e:
            logging.warning(f"Error checking CUDA runtime API: {str(e)}")
            
        logging.warning("CUDA stream priorities not supported on this system")
        return False
                
    def __init__(self, device=None, priority=None):
        """
        Create a CUDA stream with the specified priority.
        
        Args:
            device: CUDA device to use
            priority: Stream priority: PRIORITY_HIGH, PRIORITY_NORMAL, or PRIORITY_LOW
        """
        self.device = device if device is not None else torch.device('cuda')
        self.priority = priority
        
        # Check if priority is supported
        supported = self._check_priority_support()
        
        # Create stream based on priority support
        if supported and priority is not None:
            if self._priority_needs_kwargs:
                # Create with kwargs even though not in signature
                self.stream = torch.cuda.Stream(device=self.device, priority=priority)
                logging.info(f"Created CUDA stream with priority={priority} using kwargs")
            else:
                # Create with standard parameter
                try:
                    self.stream = torch.cuda.Stream(device=self.device, priority=priority)
                    logging.info(f"Created CUDA stream with priority={priority}")
                except Exception as e:
                    logging.warning(f"Failed to create prioritized stream: {str(e)}")
                    self.stream = torch.cuda.Stream(device=self.device)
        else:
            # Create regular stream without priority
            self.stream = torch.cuda.Stream(device=self.device)
            logging.info("Created regular CUDA stream (no priority)")
    
    def __enter__(self):
        """Context manager entry"""
        self._old_stream = torch.cuda.current_stream()
        torch.cuda.set_stream(self.stream)
        return self.stream
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        torch.cuda.set_stream(self._old_stream)
        
    # Note: The for_plugin method has been removed since plugins now
    # create and maintain their own prioritized CUDA streams at initialization

class ResourceManager:
    """
    Centralized resource manager for balancing GPU and CPU usage across plugins.
    Implements a singleton pattern - only one instance will exist.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize the resource manager state"""
        self.plugins = weakref.WeakValueDictionary()  # Store references to plugins without preventing garbage collection
        self.plugin_priorities = {}  # Store priority levels for each plugin
        self.plugin_resource_weights = {}  # Store resource weight for each plugin
        self.active_plugins = set()  # Currently active plugins
        
        # Configuration
        self.update_interval = 1.0  # seconds between resource checks
        self.memory_warning_threshold = 0.15  # 15% free memory triggers warnings
        self.memory_critical_threshold = 0.10  # 10% free memory triggers interventions
        
        # Plugin classifications by resource usage
        self.high_resource_plugins = set()  # Set of plugin types known to use a lot of resources
        
        # State tracking
        self.running = False
        self.last_update = 0
        self.memory_pressure = False
        self.current_memory_pressure_level = 0  # 0=none, 1=warning, 2=critical
        
        # Track active plugin priorities even when stream priorities aren't available
        self.active_plugin_priorities = {}  # Map of active plugin names to their priorities
        self.active_plugin_threads = {}     # Map of active plugin names to their thread objects
        self.cuda_stream_priorities_available = None  # Whether CUDA stream priorities are available (determined at runtime)
        
        # Debug tracking
        self.enable_debug_logging = True  # Set to True to show detailed resource management logs
        self.last_memory_log = 0  # Last time we logged memory status
        self.memory_log_interval = 5.0  # Log memory status every 5 seconds
        self.plugin_adjustments = {}  # Track which plugins were adjusted and how
        
        # Start the monitoring thread
        self.start_monitoring()
        
        # Log initialization
        logging.info("ResourceManager initialized - monitoring GPU resources for plugins")
        
    def _register_active_plugin_priority(self, plugin_name, priority):
        """
        Register the priority of an active plugin for thread/CPU-based prioritization.
        This is used as a fallback when CUDA stream priorities aren't available.
        """
        self.active_plugin_priorities[plugin_name] = priority
        
        # Check if this is the first time we're trying to set a priority
        if self.cuda_stream_priorities_available is None:
            # We'll try to determine if stream priorities are working 
            # by checking if any plugin has a non-default thread priority
            thread = threading.current_thread()
            old_priority = thread.priority if hasattr(thread, 'priority') else None
            
            try:
                # Attempt to set the thread priority based on plugin priority
                if hasattr(thread, 'priority'):
                    if priority >= 9:  # Critical - WhisperASR, VAD
                        thread.priority = 9
                    elif priority >= 7:  # High - Video, DB
                        thread.priority = 7
                    elif priority >= 5:  # Medium - WebServer, LLM
                        thread.priority = 5
                    elif priority > 0:   # Low
                        thread.priority = 3
                    else:
                        thread.priority = 1
                        
                    # Check if it worked
                    self.cuda_stream_priorities_available = False
                    if thread.priority != old_priority:
                        logging.info(f"Using thread-based priority system for {plugin_name} (priority {priority})")
                    else:
                        logging.warning(f"Failed to set thread priority for {plugin_name}, using resource management fallbacks")
                else:
                    self.cuda_stream_priorities_available = False
                    logging.warning("Thread-based priorities not supported, using resource management fallbacks")
            except Exception as e:
                self.cuda_stream_priorities_available = False
                logging.warning(f"Thread-based priorities failed: {str(e)}, using resource management fallbacks")
                
            # Store thread for potential priority adjustments later
            self.active_plugin_threads[plugin_name] = thread
    
    def register_plugin(self, plugin, priority=0, resource_weight=1.0):
        """
        Register a plugin with the resource manager.
        
        Args:
            plugin: The plugin instance to register
            priority: Plugin priority (higher = more important when resources are constrained)
            resource_weight: Relative resource weight of this plugin (higher = uses more resources)
        """
        # Detailed tracing for priority setting
        logging.warning(f"PRIORITY DEBUG: Setting plugin {plugin.name} priority to {priority}, weight to {resource_weight}")
        logging.warning(f"PRIORITY DEBUG: Plugin object type: {type(plugin).__name__}, has resource_priority attr: {hasattr(plugin, 'resource_priority')}")
        
        if hasattr(plugin, 'resource_priority'):
            logging.warning(f"PRIORITY DEBUG: Plugin's internal resource_priority: {plugin.resource_priority}")
        
        # Check if this plugin is already registered
        if plugin.name in self.plugins:
            old_priority = self.plugin_priorities.get(plugin.name, 'NOT_FOUND')
            logging.warning(f"PRIORITY OVERWRITE: Plugin {plugin.name} already registered with priority {old_priority}, overwriting with {priority}")
        
        # Store references and values
        self.plugins[plugin.name] = plugin
        self.plugin_priorities[plugin.name] = priority
        self.plugin_resource_weights[plugin.name] = resource_weight
        
        # Double-check that it was set correctly
        stored_priority = self.plugin_priorities.get(plugin.name, 'NOT SET!')
        stored_weight = self.plugin_resource_weights.get(plugin.name, 'NOT SET!')
        logging.warning(f"PRIORITY DEBUG: Stored in ResourceManager for {plugin.name}:")
        logging.warning(f"  - Priority: {stored_priority}")
        logging.warning(f"  - Weight: {stored_weight}")
        
        # Verify the count of registered plugins
        logging.warning(f"PRIORITY DEBUG: Total registered plugins: {len(self.plugins)}")
        logging.warning(f"PRIORITY DEBUG: Total priority entries: {len(self.plugin_priorities)}")
        
        # Auto-detect high resource plugins
        if any(marker in plugin.__class__.__name__ for marker in ["LLM", "VLA", "ViT", "Whisper"]):
            self.high_resource_plugins.add(plugin.name)
            
        # Determine priority group for logging
        priority_group = "background"
        if priority >= 9:
            priority_group = "critical"
        elif priority >= 7:
            priority_group = "high"
        elif priority >= 5:
            priority_group = "medium"
        elif priority > 0:
            priority_group = "low"
            
        # More detailed logging
        if self.enable_debug_logging:
            logging.info(f"ResourceManager: Registered plugin {plugin.name} (priority={priority}, " +
                       f"group={priority_group}, weight={resource_weight})")
            
            # Log the current plugins by priority groups
            self._log_plugin_priorities()
        else:
            logging.debug(f"ResourceManager: Registered plugin {plugin.name} with priority {priority}")
        
        # Add resource manager hooks to the plugin
        self._add_plugin_hooks(plugin)
        
        return plugin
        
    def _log_plugin_priorities(self):
        """Log the current plugins by priority groups"""
        # Wait until we have a few plugins to avoid spamming logs
        if len(self.plugins) < 3:
            return
            
        priority_groups = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "background": []
        }
        
        # Group plugins by priority
        for name, priority in self.plugin_priorities.items():
            if priority >= 9:
                priority_groups["critical"].append(f"{name}({priority})")
            elif priority >= 7:
                priority_groups["high"].append(f"{name}({priority})")
            elif priority >= 5:
                priority_groups["medium"].append(f"{name}({priority})")
            elif priority > 0:
                priority_groups["low"].append(f"{name}({priority})")
            else:
                priority_groups["background"].append(f"{name}({priority})")
        
        # Format groups into a readable string
        group_str = "Resource priority groups:\n"
        for group, plugins in priority_groups.items():
            if plugins:
                group_str += f"  {group.upper()}: {', '.join(plugins)}\n"
        
        logging.info(group_str)
    
    def _add_plugin_hooks(self, plugin):
        """Add resource management hooks to a plugin"""
        # Store original process method
        if not hasattr(plugin, '_original_process'):
            plugin._original_process = plugin.process
            
            # Override process method to track activity and use prioritized CUDA streams
            def resource_managed_process(self_plugin, *args, **kwargs):
                # Notify resource manager that plugin is active
                manager = ResourceManager()
                manager.on_plugin_active(self_plugin)
                
                # Get priority for this plugin
                stored_priority = manager.plugin_priorities.get(self_plugin.name, 0)
                plugin_internal_priority = getattr(self_plugin, 'resource_priority', 'NOT_FOUND')
                
                # Detailed debug logging
                if manager.enable_debug_logging:
                    logging.warning(f"PRIORITY PROCESS: Plugin {self_plugin.name} process() called")
                    logging.warning(f"  - ResourceManager priority: {stored_priority}")
                    logging.warning(f"  - Plugin's internal resource_priority: {plugin_internal_priority}")
                
                # Apply CPU-based throttling for low-priority plugins under memory pressure
                # This helps ensure high-priority plugins get more CPU time
                if manager.memory_pressure and stored_priority < 8:
                    # Get the highest priority of any active plugin
                    highest_priority = max(manager.active_plugin_priorities.values()) if manager.active_plugin_priorities else 10
                    
                    # If this plugin has much lower priority than the highest currently active,
                    # add a small delay to give high-priority plugins more CPU time
                    if highest_priority >= 9 and stored_priority <= 5:
                        # More severe delays for lower priority plugins and higher memory pressure
                        delay = 0
                        if manager.current_memory_pressure_level == 3:  # Severe pressure
                            if stored_priority <= 3:
                                delay = 0.03  # 30ms
                            elif stored_priority <= 5:
                                delay = 0.01  # 10ms
                        elif manager.current_memory_pressure_level == 2:  # Critical pressure
                            if stored_priority <= 3:
                                delay = 0.015  # 15ms
                            elif stored_priority <= 5:
                                delay = 0.005  # 5ms
                        elif manager.current_memory_pressure_level == 1:  # Warning pressure
                            if stored_priority <= 3:
                                delay = 0.005  # 5ms
                                
                        if delay > 0:
                            time.sleep(delay)
                            if manager.enable_debug_logging:
                                logging.info(f"Applied {delay*1000:.1f}ms throttling delay to {self_plugin.name} (priority {stored_priority}) - high-priority plugins active")
                
                # Use the plugin's persistent CUDA stream if it has one
                if torch.cuda.is_available() and hasattr(torch.cuda, 'current_stream'):
                    # Check if the plugin has a CUDA stream (created at plugin init time)
                    if hasattr(self_plugin, 'cuda_stream') and self_plugin.cuda_stream is not None:
                        try:
                            # Get the plugin's stream
                            plugin_stream = self_plugin.cuda_stream
                            current_stream = torch.cuda.current_stream()
                            
                            # Use the plugin's stream for this operation
                            torch.cuda.set_stream(plugin_stream)
                            
                            # Debug logging for high-priority plugins
                            if manager.plugin_priorities.get(self_plugin.name, 0) >= 8 and manager.enable_debug_logging:
                                logging.info(f"Using plugin's prioritized CUDA stream for {self_plugin.name}")
                            
                            try:
                                # Call the original process method
                                result = self_plugin._original_process(*args, **kwargs)
                                
                                # Restore the original stream
                                torch.cuda.set_stream(current_stream)
                                return result
                            except Exception as e:
                                # Make sure we restore the stream even if there's an error
                                torch.cuda.set_stream(current_stream)
                                raise
                                
                        except Exception as e:
                            # If something goes wrong with using the plugin's stream,
                            # log it and fall back to normal processing
                            logging.warning(f"Error using plugin's CUDA stream for {self_plugin.name}: {str(e)}")
                            logging.warning("Falling back to default stream")
                            # Continue to the default processing below
                
                # If we didn't use a prioritized stream, just call the original method
                result = self_plugin._original_process(*args, **kwargs)
                
                # Notify resource manager that plugin is now inactive
                manager.on_plugin_inactive(self_plugin)
                
                return result
            
            # Replace the process method
            plugin.process = resource_managed_process.__get__(plugin)
            
        # Add resource management methods if they don't exist
        if not hasattr(plugin, 'adjust_for_resource_constraints'):
            def adjust_for_resource_constraints(self_plugin, pressure_level):
                """Adjust plugin behavior based on resource pressure level"""
                if hasattr(self_plugin, 'batch_size') and pressure_level > 0:
                    # Reduce batch size under pressure
                    original_batch = getattr(self_plugin, '_original_batch_size', self_plugin.batch_size)
                    if not hasattr(self_plugin, '_original_batch_size'):
                        self_plugin._original_batch_size = original_batch
                    
                    # Adjust based on pressure level
                    if pressure_level == 1:  # Warning
                        self_plugin.batch_size = max(1, int(original_batch * 0.7))
                    elif pressure_level >= 2:  # Critical
                        self_plugin.batch_size = max(1, int(original_batch * 0.5))
                        
                if hasattr(self_plugin, 'max_new_tokens') and pressure_level > 0:
                    # Reduce token generation under pressure
                    original_tokens = getattr(self_plugin, '_original_max_tokens', self_plugin.max_new_tokens)
                    if not hasattr(self_plugin, '_original_max_tokens'):
                        self_plugin._original_max_tokens = original_tokens
                    
                    # Adjust based on pressure level
                    if pressure_level == 1:  # Warning
                        self_plugin.max_new_tokens = max(32, int(original_tokens * 0.7))
                    elif pressure_level >= 2:  # Critical
                        self_plugin.max_new_tokens = max(32, int(original_tokens * 0.5))
                
                return True
            
            plugin.adjust_for_resource_constraints = adjust_for_resource_constraints.__get__(plugin)
            
        # Add memory release method if it doesn't exist
        if not hasattr(plugin, 'release_memory'):
            def release_memory(self_plugin, pressure_level):
                """Release memory based on pressure level"""
                if pressure_level <= 0:
                    return False  # No need to release
                
                # Clear any caches
                for attr_name in dir(self_plugin):
                    if 'cache' in attr_name.lower() and isinstance(getattr(self_plugin, attr_name), dict):
                        cache = getattr(self_plugin, attr_name)
                        if pressure_level == 1:  # Warning - partial clear
                            # Keep essential items
                            to_remove = [k for k in cache.keys() if not k.startswith('essential_')]
                            for key in to_remove:
                                del cache[key]
                        elif pressure_level >= 2:  # Critical - full clear
                            cache.clear()
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return True
            
            plugin.release_memory = release_memory.__get__(plugin)
    
    def on_plugin_active(self, plugin):
        """Called when a plugin becomes active"""
        plugin_name = getattr(plugin, 'name', str(plugin))
        self.active_plugins.add(plugin_name)
        
        # Store the plugin's priority in active_plugin_priorities
        priority = self.plugin_priorities.get(plugin_name, 0)
        self.active_plugin_priorities[plugin_name] = priority
        
        # Enhanced debug logging for active plugins with priority inspection
        if self.enable_debug_logging:
            plugin_internal_priority = getattr(plugin, 'resource_priority', 'NOT_FOUND')
            plugin_registered = plugin_name in self.plugin_priorities
            
            logging.info(f"Plugin {plugin_name} became active with priority {priority}")
            
            # More detailed logging at debug level
            if plugin.name in ['WhisperASR', 'VADFilter', 'NanoLLM', 'VideoSource', 'VideoOutput'] or plugin_internal_priority != priority:
                logging.warning(f"PRIORITY VALUES: {plugin_name} has:")
                logging.warning(f"  - ResourceManager stored priority: {priority}")
                logging.warning(f"  - Internal resource_priority attr: {plugin_internal_priority}")
                logging.warning(f"  - Is registered with ResourceManager: {plugin_registered}")
                
                # Log all active plugins and their priorities
                active_plugins_with_priority = [(name, self.active_plugin_priorities.get(name, 0)) 
                                             for name in self.active_plugins]
                active_plugins_with_priority.sort(key=lambda x: -x[1])  # Sort by priority (highest first)
                logging.warning(f"Current active plugins: {active_plugins_with_priority}")
        
        # If high resource plugin becomes active, check if we need to notify other plugins
        if plugin_name in self.high_resource_plugins and self.memory_pressure:
            self.balance_resources()
    
    def on_plugin_inactive(self, plugin):
        """Called when a plugin becomes inactive"""
        plugin_name = getattr(plugin, 'name', str(plugin))
        
        # Remove from active sets
        if plugin_name in self.active_plugins:
            self.active_plugins.remove(plugin_name)
            
        if plugin_name in self.active_plugin_priorities:
            del self.active_plugin_priorities[plugin_name]
            
        if plugin_name in self.active_plugin_threads:
            del self.active_plugin_threads[plugin_name]
    
    def start_monitoring(self):
        """Start the resource monitoring thread"""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop the resource monitoring thread"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_resources(self):
        """Background thread to monitor system resources"""
        while self.running:
            current_time = time.time()
            
            # Only check periodically
            if current_time - self.last_update >= self.update_interval:
                self.last_update = current_time
                self.check_memory_pressure()
            
            time.sleep(0.1)  # Sleep to avoid busy-waiting
    
    def check_memory_pressure(self):
        """Check if we're under memory pressure and update state"""
        if not torch.cuda.is_available():
            return
            
        # Get current memory stats
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_percent = free_mem / total_mem
        used_mem = total_mem - free_mem
        used_percent = used_mem / total_mem
        
        # Periodically log memory usage stats if debug logging is enabled
        current_time = time.time()
        if self.enable_debug_logging and current_time - self.last_memory_log >= self.memory_log_interval:
            self.last_memory_log = current_time
            
            # Make it visually distinct in the logs
            if used_percent > 0.85:  # > 85% used
                logging.warning(f"GPU Memory: {used_mem/1024/1024:.1f}MB / {total_mem/1024/1024:.1f}MB " +
                             f"({used_percent:.1%} used, {free_percent:.1%} free)")
            else:
                logging.info(f"GPU Memory: {used_mem/1024/1024:.1f}MB / {total_mem/1024/1024:.1f}MB " +
                           f"({used_percent:.1%} used, {free_percent:.1%} free)")
                
            # Log active plugins
            if self.active_plugins:
                # Sort by priority
                active_with_priority = [(name, self.plugin_priorities.get(name, 0)) 
                                     for name in self.active_plugins]
                active_with_priority.sort(key=lambda x: -x[1])  # Sort by priority (highest first)
                
                active_plugins_str = ", ".join([f"{name}({prio})" for name, prio in active_with_priority])
                logging.info(f"Active plugins: {active_plugins_str}")
        
        old_pressure_level = self.current_memory_pressure_level
        
        # Update pressure state with more granular levels
        if free_percent < self.memory_critical_threshold * 0.5:  # < 5% free
            self.memory_pressure = True
            self.current_memory_pressure_level = 3  # Severe
        elif free_percent < self.memory_critical_threshold:  # < 10% free
            self.memory_pressure = True
            self.current_memory_pressure_level = 2  # Critical
        elif free_percent < self.memory_warning_threshold:  # < 15% free
            self.memory_pressure = True
            self.current_memory_pressure_level = 1  # Warning
        else:
            self.memory_pressure = False
            self.current_memory_pressure_level = 0  # Normal
        
        # If pressure level changed, balance resources
        if old_pressure_level != self.current_memory_pressure_level:
            # Get pressure level name for logging
            pressure_name = "NORMAL"
            if self.current_memory_pressure_level == 1:
                pressure_name = "WARNING"
            elif self.current_memory_pressure_level == 2:
                pressure_name = "CRITICAL"
            elif self.current_memory_pressure_level == 3:
                pressure_name = "SEVERE"
                
            if self.current_memory_pressure_level > old_pressure_level:
                logging.warning(f"MEMORY PRESSURE INCREASED TO {pressure_name} (level {self.current_memory_pressure_level}) - " +
                             f"{free_percent:.1%} free, {used_percent:.1%} used")
            else:
                old_name = "NORMAL"
                if old_pressure_level == 1:
                    old_name = "WARNING"
                elif old_pressure_level == 2:
                    old_name = "CRITICAL"
                elif old_pressure_level == 3:
                    old_name = "SEVERE"
                    
                logging.info(f"Memory pressure decreased from {old_name} to {pressure_name} - " +
                           f"{free_percent:.1%} free, {used_percent:.1%} used")
                
            # Reset adjustment tracking on pressure level change
            self.plugin_adjustments = {}
            
            # Balance resources
            self.balance_resources()
    
    def balance_resources(self):
        """
        Balance resources among plugins when under pressure
        This is where the core resource balancing logic happens
        """
        if not self.memory_pressure or len(self.plugins) == 0:
            return
            
        # Get active plugins sorted by priority (highest first)
        active_plugins = [(name, self.plugin_priorities.get(name, 0)) 
                         for name in self.active_plugins]
        active_plugins.sort(key=lambda x: -x[1])  # Sort by priority (highest first)
        
        # Create priority groups for more granular control
        priority_groups = {
            "critical": [],   # WhisperASR (10)
            "high": [],       # VideoSource/VideoOutput (8), NanoDB (7)
            "medium": [],     # WebServer/EventFilter (6), LLM (5)
            "low": [],        # NanoVLA (3)
            "background": []  # Everything else
        }
        
        # Log raw active plugin data before categorization
        logging.warning(f"PRIORITY BALANCE: Raw active plugins with priorities before grouping:")
        for name, priority in active_plugins:
            plugin_obj = self.plugins.get(name)
            if plugin_obj:
                internal_priority = getattr(plugin_obj, 'resource_priority', 'NOT_FOUND')
                logging.warning(f"  - {name}: ResourceManager priority={priority}, internal priority={internal_priority}")
        
        # Categorize plugins based on priority
        for name, priority in active_plugins:
            if priority >= 9:
                priority_groups["critical"].append(name)
            elif priority >= 7:
                priority_groups["high"].append(name)
            elif priority >= 5:
                priority_groups["medium"].append(name)
            elif priority > 0:
                priority_groups["low"].append(name)
            else:
                priority_groups["background"].append(name)
                
        # Log priority groups for debugging
        logging.debug(f"Resource priority groups: {priority_groups}")
        
        # Log which plugins will be adjusted
        if self.enable_debug_logging:
            if self.current_memory_pressure_level == 1:
                logging.info("Resource adjustments: Applying constraints to LOW and BACKGROUND priority plugins")
            elif self.current_memory_pressure_level == 2: 
                logging.info("Resource adjustments: Applying constraints to HIGH, MEDIUM, LOW and BACKGROUND priority plugins")
            elif self.current_memory_pressure_level == 3:
                logging.warning("Resource adjustments: Applying strong constraints to all non-CRITICAL priority plugins")
                
        # Adjust behavior based on pressure level and priority group
        if self.current_memory_pressure_level == 1:  # Warning level
            # Only adjust low and background priority plugins
            for group in ["low", "background"]:
                for name in priority_groups[group]:
                    plugin = self.plugins.get(name)
                    if plugin is not None and hasattr(plugin, 'adjust_for_resource_constraints'):
                        result = plugin.adjust_for_resource_constraints(self.current_memory_pressure_level)
                        if result and self.enable_debug_logging:
                            self.plugin_adjustments[name] = self.current_memory_pressure_level
                            logging.info(f"Adjusted plugin {name} for WARNING pressure")
        else:  # Critical/Severe level
            # Adjust all but critical priority plugins
            for group in ["high", "medium", "low", "background"]:
                for name in priority_groups[group]:
                    plugin = self.plugins.get(name)
                    if plugin is not None and hasattr(plugin, 'adjust_for_resource_constraints'):
                        # Vary adjustment level based on group and pressure level
                        if group == "high":
                            adjustment_level = 1  # High priority plugins get milder adjustments
                        elif group == "medium" and self.current_memory_pressure_level == 2:
                            adjustment_level = 1  # Medium plugins get milder adjustments in critical
                        else:
                            adjustment_level = self.current_memory_pressure_level
                            
                        result = plugin.adjust_for_resource_constraints(adjustment_level)
                        if result and self.enable_debug_logging:
                            self.plugin_adjustments[name] = adjustment_level
                            pressure_name = "CRITICAL" if self.current_memory_pressure_level == 2 else "SEVERE"
                            logging.info(f"Adjusted plugin {name} (level {adjustment_level}) for {pressure_name} pressure")
        
        # If we're in critical pressure, ask low-priority plugins to release memory
        if self.current_memory_pressure_level >= 2:
            # Log memory release actions
            if self.enable_debug_logging:
                if self.current_memory_pressure_level == 2:
                    logging.info("Memory release: Requesting LOW and BACKGROUND priority plugins to release memory")
                else:
                    logging.warning("Memory release: Requesting LOW, BACKGROUND, and MEDIUM priority plugins to release memory")
            
            # First ask background and low priority plugins to release memory
            for group in ["background", "low"]:
                for name in priority_groups[group]:
                    plugin = self.plugins.get(name)
                    if plugin is not None and hasattr(plugin, 'release_memory'):
                        result = plugin.release_memory(self.current_memory_pressure_level)
                        if result and self.enable_debug_logging:
                            pressure_name = "CRITICAL" if self.current_memory_pressure_level == 2 else "SEVERE"
                            logging.info(f"Released memory from {name} for {pressure_name} pressure")
            
            # If still in critical pressure, ask medium priority plugins too
            if self.current_memory_pressure_level >= 3:
                for name in priority_groups["medium"]:
                    plugin = self.plugins.get(name)
                    if plugin is not None and hasattr(plugin, 'release_memory'):
                        result = plugin.release_memory(1)  # Use lower pressure level for medium priority
                        if result and self.enable_debug_logging:
                            logging.info(f"Released memory from {name} for SEVERE pressure (level 1)")
                            
            # Log summary of memory release
            if self.enable_debug_logging and len(self.plugin_adjustments) > 0:
                affected_plugins = len(self.plugin_adjustments)
                logging.info(f"Resource balancing applied to {affected_plugins} plugins")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()