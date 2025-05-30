# Image Sub-extractor

A Python application that detects and extracts multiple sub-images from a single JPG file using OpenCV. Ideal for processing scanned documents and scrapbook pages.

## Features

- Detects distinct images within a larger image
- Automatically crops and extracts each detected image
- Saves extracted images as separate JPG files
- Adds padding around extracted images to ensure no content is cut off
- Two specialized processing modes:
  - Standard mode for regular scanned documents
  - Scrapbook mode optimized for colored background pages
- Debug visualization mode to see each processing step
- Supports batch processing of multiple images

## Requirements

- Python 3.x
- OpenCV (opencv-python) >= 4.8.0
- NumPy >= 1.24.0

## Installation

1. Create a virtual environment (recommended):

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

2. Install required packages:

    ```powershell
    pip install -r requirements.txt
    ```

## Usage

The script can process either a single image or a folder of images.

### Command Line Options

```powershell
python image_extractor.py [-h] [--input INPUT] [--output OUTPUT] [--debug] [--scrapbook]
```

Options:

- `-i, --input`: Path to input image or folder
- `-o, --output`: Path to output directory
- `-d, --debug`: Enable debug visualization mode
- `-s, --scrapbook`: Enable scrapbook mode for colored background pages

### Examples

Process a single image in standard mode:

```powershell
python image_extractor.py -i path/to/image.jpg -o path/to/output
```

Process a folder with debug visualization:

```powershell
python image_extractor.py -i path/to/folder -o path/to/output -d
```

Process scrapbook pages:

```powershell
python image_extractor.py -i path/to/scrapbook.jpg -o path/to/output -s
```

If no paths are provided, the program will prompt you for:

1. The path to your input image or folder
2. The output directory where extracted images should be saved

### Output Structure

When processing a folder:

- A subfolder will be created for each input image
- Extracted images will be named `{original_name}_image_{n}.jpg`
- Debug files (if enabled) will include:
  - `{original_name}_debug_binary.jpg`: Binary threshold result
  - `{original_name}_debug_morph.jpg`: Morphological operations
  - `{original_name}_debug_detection.jpg`: Final detection visualization
- Supported input formats: .jpg, .jpeg, .png, .bmp

## How it Works

1. The script uses OpenCV to read and process the input image
2. Based on the mode (standard/scrapbook):
   - Standard: Converts to grayscale and enhances contrast
   - Scrapbook: Uses LAB color space for better color separation
3. Applies edge detection and adaptive thresholding
4. Uses morphological operations to connect edges
5. Detects contours and filters them based on size
6. Groups nearby contours to handle split detections
7. Extracts each detected region with padding
8. Saves individual images as separate files

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
