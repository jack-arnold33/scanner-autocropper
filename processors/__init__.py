"""
Package initialization for image processors.
"""

from .base_processor import BaseImageProcessor
from .standard_processor import StandardImageProcessor
from .scrapbook_processor import ScrapbookImageProcessor

__all__ = ['BaseImageProcessor', 'StandardImageProcessor', 'ScrapbookImageProcessor']
