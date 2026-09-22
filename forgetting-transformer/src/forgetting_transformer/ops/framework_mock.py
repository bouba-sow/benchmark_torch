"""
Mock framework module for ndr geometric attention
只保留必要的部分
"""
import torch
from typing import Optional, Any

class visualize:
    """Mock visualize class"""
    @staticmethod
    def attention(*args, **kwargs):
        """Dummy attention visualization"""
        pass
    
    @staticmethod
    def plot(*args, **kwargs):
        """Dummy plot"""
        pass

# Mock其他可能需要的功能
def get_logger(name: str):
    """Mock logger"""
    import logging
    return logging.getLogger(name)

