import os
import argparse
from pathlib import Path
from processors import StandardImageProcessor, ScrapbookImageProcessor

def process_folder(input_folder, output_dir, debug=False, scrapbook_mode=False, rotation=None):
    """Process all images in the input folder and save extracted sub-images to output directory.
    
    Args:
        input_folder (str): Path to the folder containing input images
        output_dir (str): Directory to save extracted images
        debug (bool): If True, shows visualization of each processing step
        scrapbook_mode (bool): If True, uses the scrapbook image processor
        rotation (str, optional): Direction to rotate images. Can be 'left' or 'right'
    """
    # Create the appropriate processor
    processor = ScrapbookImageProcessor(debug, rotation) if scrapbook_mode else StandardImageProcessor(debug, rotation)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List all image files in the input folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    input_files = [f for f in Path(input_folder).glob('*') if f.suffix.lower() in image_extensions]
    
    if not input_files:
        processor.logger.error(f"No image files found in {input_folder}")
        return
    
    total_images = 0
    processed_files = 0
    
    for input_file in input_files:
        processor.logger.info(f"\nProcessing file {processed_files + 1}/{len(input_files)}: {input_file.name}")
        
        try:
            # Process the image using the selected processor
            saved_count = processor.process_image(str(input_file), output_dir)
            total_images += saved_count
            processed_files += 1
            processor.logger.info(f"Extracted {saved_count} images from {input_file.name}")
            
        except Exception as e:
            processor.logger.error(f"Error processing {input_file.name}: {str(e)}")
            continue
    
    processor.logger.info(f"\nProcessing complete. Processed {processed_files} files and extracted {total_images} images.")
    return total_images

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract sub-images from scanned images.')
    parser.add_argument('--input', '-i', type=str, 
                       help='Path to the input folder containing images')
    parser.add_argument('--output', '-o', type=str, 
                       help='Output directory path')
    parser.add_argument('--debug', '-d', action='store_true', 
                       help='Enable debug mode to save visualization of detection process')
    parser.add_argument('--scrapbook', '-s', action='store_true',
                       help='Enable scrapbook mode for processing scrapbook page images')
    parser.add_argument('--rotate', '-r', choices=['left', 'right'],
                       help='Rotate output images 90 degrees left or right')
    args = parser.parse_args()
    
    # Get input and output paths
    input_path = args.input
    output_dir = args.output
    
    # If paths not provided via arguments, prompt user
    if not input_path:
        input_path = input("Enter the path to your input folder: ").strip().strip('"\'')
    if not output_dir:
        output_dir = input("Enter the output directory path: ").strip().strip('"\'')
        
    # Convert to Path objects for proper path handling
    try:
        input_path = Path(input_path.strip().strip('"\''))
        output_dir = Path(output_dir.strip().strip('"\''))
        
        # Validate input path exists
        if not input_path.exists():
            print(f"Error: Input path does not exist: {input_path}")
            return
              # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_saved = process_folder(
            str(input_path),
            str(output_dir),
            debug=args.debug,
            scrapbook_mode=args.scrapbook,
            rotation=args.rotate
        )
        
        if total_saved:
            print(f"\nSuccessfully extracted {total_saved} images to {output_dir}")
        else:
            print("No images were extracted")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
