import cv2
import logging
from pathlib import Path
import abc

class BaseImageProcessor(abc.ABC):
    """Base class for image processors."""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.logger = self._setup_logging()
        self._windows = []  # Track open windows
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(level=log_level,
                          format='%(asctime)s - %(levelname)s - %(message)s',
                          datefmt='%H:%M:%S')
        return logging.getLogger(__name__)
    
    def debug_step(self, name, image, wait=False):
        """Helper function to handle debug visualization of each processing step."""
        if self.debug:
            # Create and position window
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            window_width = 800  # Increased window size
            window_height = 600
            cols = 3
            row = (int(name.split('.')[0]) - 1) // cols
            col = (int(name.split('.')[0]) - 1) % cols
            x_pos = col * window_width
            y_pos = row * window_height
            cv2.moveWindow(name, x_pos, y_pos)
            
            # Show and resize image
            cv2.imshow(name, image)
            cv2.resizeWindow(name, window_width, window_height)
            
            # Track this window
            self._windows.append(name)
              # Print step name
            print(f"\nShowing step: {name}")
              # Only wait for key press on the final step, otherwise brief pause
            if name.startswith("9."):
                print("\nProcessing complete. Press any key to close all windows...")
                cv2.waitKey(0)
                # Clean up all windows
                for win in self._windows:
                    cv2.destroyWindow(win)
                self._windows = []
            else:
                cv2.waitKey(1)  # Brief pause to allow window to render
                
        return image
    
    @abc.abstractmethod
    def process_image(self, image_path, output_dir):
        """Process a single image and extract sub-images.
        
        Args:
            image_path (str): Path to the input image
            output_dir (str): Directory to save extracted images
            
        Returns:
            int: Number of images extracted
        """
        pass
