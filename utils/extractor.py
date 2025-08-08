# utils/extractor.py

"""
extractor.py

Handles layout-aware data extraction from PDF files. Uses pdfplumber to extract:
- Text tokens (words)
- Bounding boxes (absolute)
- Page-level images

Returns:
    - List of text tokens
    - List of bounding boxes
    - List of page numbers (aligned with tokens)
    - List of PIL page images

Used as the first step in the semantic field disambiguation pipeline.
"""

import pdfplumber
from PIL import Image

def extract_layout_data(pdf_path):
    """
    Extracts tokens, bounding boxes, page numbers, and page images from a PDF.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        Tuple[List[str], List[Tuple[int, int, int, int]], List[int], List[Image.Image]]:
            - tokens: Text tokens
            - bboxes: Bounding boxes (x0, y0, x1, y1)
            - page_nums: Page numbers per token (1-based index)
            - page_images: PIL images of each page
    """
    tokens = []
    bboxes = []
    page_nums = []
    page_images = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_number = i + 1

            # Render the page as image
            image = page.to_image(resolution=150).original.convert("RGB")
            page_images.append(image)

            # Extract words and bounding boxes
            words = page.extract_words()
            for word in words:
                tokens.append(word["text"])
                bbox = (word["x0"], word["top"], word["x1"], word["bottom"])
                bboxes.append(bbox)
                page_nums.append(page_number)

    return tokens, bboxes, page_nums, page_images
