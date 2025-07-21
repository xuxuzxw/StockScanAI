"""
Error handling utilities for system checks
"""
import functools
import logging
import time
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


def safe_execute(func: Callable, *args, **kwargs) -> tuple[bool, Any]:
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        tuple: (success: bool, result: Any)
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        log.error(f"Function {func.__name__} failed: {e}")
        return False, None


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry function execution on failure
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        log.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        time.sleep(delay)
                    else:
                        log.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator


def timeout_handler(timeout_seconds: int = 30):
    """
    Decorator to add timeout handling to functions
    
    Args:
        timeout_seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds")
            
            # Set the timeout handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel the alarm
                return result
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        
        return wrapper
    return decorator


class CheckResult:
    """Represents the result of a system check"""
    
    def __init__(self, name: str, success: bool, message: str = "", 
                 duration: float = 0.0, error: Optional[Exception] = None):
        self.name = name
        self.success = success
        self.message = message
        self.duration = duration
        self.error = error
        self.timestamp = time.time()
    
    def to_dict(self) -> dict:
        """Convert result to dictionary"""
        return {
            'name': self.name,
            'success': self.success,
            'message': self.message,
            'duration': self.duration,
            'error': str(self.error) if self.error else None,
            'timestamp': self.timestamp
        }
    
    def __str__(self) -> str:
        status = "PASS" if self.success else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


def measure_execution_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            log.debug(f"{func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            log.error(f"{func.__name__} failed after {duration:.3f}s: {e}")
            raise
    return wrapper