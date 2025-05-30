import cv2
import numpy as np
import os
from pathlib import Path
from .base_processor import BaseImageProcessor

class ScrapbookImageProcessor(BaseImageProcessor):
    """Processor optimized for scrapbook page images with colored backgrounds."""
    
    def __init__(self, debug=False):
        super().__init__(debug)
        self.rotation_threshold = 30  # Max rotation angle to correct (degrees)
        self.min_line_length = 100   # Minimum length for line detection
    
    def process_image(self, image_path, output_dir):
        if self.debug:
            print("\nDebug mode: Each processing step will be shown in a separate window.")
            print("Windows are resizable and can be arranged for better viewing.")
            print("Steps will appear one by one, and all will remain visible.")
            print("Press any key after each step appears to continue...\n")
        
        # Get source image name without extension
        source_name = Path(image_path).stem
        
        # Read the image
        self.logger.info(f"Reading image from {image_path}")
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Show original image in debug mode
        self.debug_step("1. Original Image", original_image, wait=False)
        
        # Get image dimensions
        height, width = original_image.shape[:2]
        self.logger.info(f"Original image dimensions: {width}x{height}")
        
        # Convert to LAB color space for better color separation
        lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        self.debug_step("2. LAB Channels", np.hstack([l_channel, a_channel, b_channel]), wait=False)
        
        # Estimate background color from corners
        corner_size = 50
        corners = [
            original_image[0:corner_size, 0:corner_size],
            original_image[0:corner_size, -corner_size:],
            original_image[-corner_size:, 0:corner_size],
            original_image[-corner_size:, -corner_size:]
        ]
        bg_color = np.median([np.median(corner, axis=(0,1)) for corner in corners], axis=0)
        self.logger.info(f"Estimated background color (BGR): {bg_color}")
        
        # Create a background difference mask
        diff_image = np.zeros_like(original_image)
        for i in range(3):
            diff_image[:,:,i] = cv2.absdiff(original_image[:,:,i], int(bg_color[i]))
        diff_gray = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
        self.debug_step("3. Background Difference", diff_gray, wait=False)
        
        # Edge detection on L channel with automatic thresholds
        edges_canny = cv2.Canny(l_channel, 50, 150, apertureSize=3)
        
        # Sobel edge detection for additional edge information
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel * 255 / np.max(sobel))
        
        # Combine different edge detection methods
        edges = cv2.addWeighted(edges_canny, 0.7, sobel, 0.3, 0)
        self.debug_step("4. Enhanced Edge Detection", edges, wait=False)
        
        # Detect lines for rotation correction
        rotated_image = self._correct_rotation(original_image, edges)
        if rotated_image is not original_image:  # Image was rotated
            # Update edge detection on rotated image
            lab_rotated = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2LAB)
            l_channel_rotated = cv2.split(lab_rotated)[0]
            edges = cv2.Canny(l_channel_rotated, 50, 150, apertureSize=3)
            self.debug_step("4b. Edges After Rotation", edges, wait=False)
        
        # Combine edge detection with background difference
        combined = cv2.addWeighted(diff_gray, 0.6, edges, 0.4, 0)
        
        # Apply adaptive thresholding instead of global Otsu
        binary = cv2.adaptiveThreshold(
            combined, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            101,  # Block size
            2    # C constant
        )
        self.debug_step("5. Combined Binary", binary, wait=False)
        
        # More aggressive morphological operations
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # First remove small noise
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, square_kernel, iterations=1)
        
        # Then connect edges
        dilated = cv2.dilate(denoised, rect_kernel, iterations=2)
        self.debug_step("6. Dilated Binary", dilated, wait=False)
        
        # Close gaps while preserving shape
        morph = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, rect_kernel, iterations=2)
        self.debug_step("7. Morphological Operations", morph, wait=False)
        
        # Find and filter contours
        contours = self._find_contours(morph, rotated_image)  # Use rotated image
        valid_contours = self._filter_contours(contours, width, height)
        
        # Extract and save images
        saved_images = self._save_extracted_images(
            rotated_image, valid_contours, output_dir, source_name
        )
        
        if self.debug:
            self._save_debug_images(rotated_image, binary, morph, dilated, 
                                  valid_contours, output_dir, source_name)
        
        return saved_images
    
    def _find_contours(self, morph, original_image):
        """Find contours in the processed image."""
        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if self.debug:
            contour_vis = original_image.copy()
            cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
            self.debug_step("8. Initial Contours", contour_vis, wait=True)
            
        self.logger.info(f"Found {len(contours)} initial contours")
        return contours
    
    def _filter_contours(self, contours, width, height):
        """Filter and group contours based on size and proximity."""
        potential_rects = []
        total_area = width * height
        # More lenient area thresholds for scrapbook pages
        min_area_threshold = total_area * 0.03  # 3% of total image area
        max_area_threshold = total_area * 0.75  # 75% of total image area
        
        for contour in contours:
            # Use less approximation for more accurate boundaries
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            
            if area < min_area_threshold:
                self.logger.debug(f"Rejected rectangle {w}x{h} - too small ({(area/total_area)*100:.1f}% of image)")
                continue
            if area > max_area_threshold:
                self.logger.debug(f"Rejected rectangle {w}x{h} - too large ({(area/total_area)*100:.1f}% of image)")
                continue
                
            self.logger.debug(f"Accepted rectangle {w}x{h} ({(area/total_area)*100:.1f}% of image)")
            potential_rects.append((x, y, w, h))
        
        self.logger.info(f"Found {len(potential_rects)} potential rectangles before grouping")
        valid_contours = self._group_rectangles(potential_rects)
        self.logger.info(f"Found {len(valid_contours)} grouped rectangles")
        
        return valid_contours
    
    def _group_rectangles(self, rectangles, threshold_distance=80):
        """Group nearby rectangles into a single bounding box."""
        if not rectangles:
            return []
        
        groups = []
        used = set()
        
        for i, (x1, y1, w1, h1) in enumerate(rectangles):
            if i in used:
                continue
                
            current_group = [(x1, y1, w1, h1)]
            used.add(i)
            
            for j, (x2, y2, w2, h2) in enumerate(rectangles[i+1:], i+1):
                if j in used:
                    continue
                    
                distance_x = min(abs(x1 - x2), abs(x1 + w1 - (x2 + w2)))
                distance_y = min(abs(y1 - y2), abs(y1 + h1 - (y2 + h2)))
                
                if distance_x < threshold_distance and distance_y < threshold_distance:
                    current_group.append((x2, y2, w2, h2))
                    used.add(j)
            
            if current_group:
                x = min(rect[0] for rect in current_group)
                y = min(rect[1] for rect in current_group)
                max_x = max(rect[0] + rect[2] for rect in current_group)
                max_y = max(rect[1] + rect[3] for rect in current_group)
                groups.append((x, y, max_x - x, max_y - y))
        
        return groups
    
    def _save_extracted_images(self, original_image, valid_contours, output_dir, source_name):
        """Extract and save sub-images from the original image."""
        saved_images = 0
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (x, y, w, h) in enumerate(valid_contours):
            self.logger.debug(f"Processing rectangle {w}x{h} at ({x},{y})")
            
            try:
                # Add small padding around extracted images
                pad = 5
                y1 = max(0, y - pad)
                y2 = min(original_image.shape[0], y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(original_image.shape[1], x + w + pad)
                
                sub_image = original_image[y1:y2, x1:x2]
                
                if sub_image.size == 0:
                    self.logger.warning(f"Skipping empty sub-image at coordinates ({x}, {y}, {w}, {h})")
                    continue
                    
                # Apply rotation if specified
                sub_image = self.rotate_image(sub_image)
                
                output_path = os.path.join(output_dir, f'{source_name}_image_{saved_images + 1}.jpg')
                success = cv2.imwrite(output_path, sub_image)
                
                if success:
                    saved_images += 1
                    self.logger.info(f"Saved sub-image to {output_path}")
                else:
                    self.logger.error(f"Failed to save sub-image to {output_path}")
                    
            except Exception as e:
                self.logger.error(f"Error processing sub-image: {str(e)}")
                
        return saved_images
    
    def _save_debug_images(self, original_image, binary, morph, dilated, 
                          valid_contours, output_dir, source_name):
        """Save debug visualization images."""
        debug_image = original_image.copy()
        for i, (x, y, w, h) in enumerate(valid_contours):
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_image, f'#{i + 1}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                       
        cv2.imwrite(os.path.join(output_dir, f'{source_name}_debug_binary.jpg'), binary)
        cv2.imwrite(os.path.join(output_dir, f'{source_name}_debug_morph.jpg'), morph)
        cv2.imwrite(os.path.join(output_dir, f'{source_name}_debug_dilated.jpg'), dilated)
        cv2.imwrite(os.path.join(output_dir, f'{source_name}_debug_detection.jpg'), debug_image)
        
        final_vis = original_image.copy()
        for i, (x, y, w, h) in enumerate(valid_contours):
            cv2.rectangle(final_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(final_vis, f'#{i + 1}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                       
        self.debug_step("9. Extracted Regions", final_vis, wait=True)
        cv2.imwrite(os.path.join(output_dir, f'{source_name}_debug_final.jpg'), final_vis)
