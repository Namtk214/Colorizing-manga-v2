#!/usr/bin/env python3
"""
Example script demonstrating PDF processing functionality
"""
import os
import sys
import argparse
from pdf_processor import PDFProcessor
from colorizator import MangaColorizator

def test_pdf_processing():
    """
    Test PDF processing with a sample PDF or demonstrate usage
    """
    print("PDF Processing Test")
    print("==================")
    
    # Check if PyMuPDF is installed
    try:
        import fitz
        print("✓ PyMuPDF (fitz) is available")
    except ImportError:
        print("✗ PyMuPDF not installed. Run: pip install PyMuPDF")
        return False
    
    # Check if PIL is available
    try:
        from PIL import Image
        print("✓ PIL (Pillow) is available")
    except ImportError:
        print("✗ PIL not installed. Run: pip install Pillow")
        return False
    
    print("\nPDF processing functionality is ready!")
    print("\nUsage examples:")
    print("="*50)
    
    print("\n1. Basic PDF processing:")
    print("   python inference.py -p manga.pdf")
    
    print("\n2. PDF with batch processing (20 pages at once):")
    print("   python inference.py -p manga.pdf -b 20")
    
    print("\n3. PDF with GPU and high DPI:")
    print("   python inference.py -p manga.pdf -b 8 -g --pdf_dpi 200")
    
    print("\n4. PDF without creating output PDF (images only):")
    print("   python inference.py -p manga.pdf --no_pdf_output")
    
    print("\nOutput structure:")
    print("- manga_colorized/")
    print("  ├── manga_colorized_page_001.png")
    print("  ├── manga_colorized_page_002.png")
    print("  ├── ...")
    print("  └── manga_colorized.pdf")
    
    print("\nParameters:")
    print("- --pdf_dpi: Resolution for conversion (default: 150)")
    print("  * 150: Good quality, reasonable file size")
    print("  * 300: High quality, larger files")
    print("  * 72:  Low quality, small files")
    
    print("- -b/--batch_size: Pages processed simultaneously")
    print("  * Higher = faster but more memory usage")
    print("  * Recommended: 4-20 depending on your GPU/RAM")
    
    return True

def create_sample_pdf():
    """
    Create a simple test PDF for demonstration
    """
    try:
        import fitz
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Create a simple test PDF
        doc = fitz.open()
        
        for i in range(3):
            page = doc.new_page(width=400, height=600)
            
            # Create a simple manga-like image
            img = Image.new('RGB', (400, 600), 'white')
            draw = ImageDraw.Draw(img)
            
            # Draw some simple manga-like content
            draw.rectangle([50, 50, 350, 150], outline='black', width=3)
            draw.text((60, 80), f"Page {i+1}", fill='black')
            draw.text((60, 110), "Sample manga content", fill='black')
            
            # Add some simple drawings
            draw.ellipse([100, 200, 200, 300], outline='black', width=2)
            draw.rectangle([150, 350, 250, 450], outline='black', width=2)
            
            # Convert PIL to bytes and insert into PDF
            img_bytes = fitz.io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            
            img_rect = fitz.Rect(0, 0, 400, 600)
            page.insert_image(img_rect, stream=img_bytes)
        
        # Save test PDF
        test_pdf_path = "test_manga.pdf"
        doc.save(test_pdf_path)
        doc.close()
        
        print(f"Created test PDF: {test_pdf_path}")
        return test_pdf_path
        
    except Exception as e:
        print(f"Could not create test PDF: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="PDF processing example and test")
    parser.add_argument("--create_test", action="store_true", help="Create a test PDF")
    parser.add_argument("--test_pdf", type=str, help="Test with specific PDF file")
    args = parser.parse_args()
    
    if args.create_test:
        pdf_path = create_sample_pdf()
        if pdf_path:
            print(f"\nNow you can test with: python inference.py -p {pdf_path}")
    
    elif args.test_pdf:
        if os.path.exists(args.test_pdf):
            print(f"Testing PDF processing with: {args.test_pdf}")
            # You could add actual testing here
            processor = PDFProcessor()
            try:
                images, info = processor.pdf_to_images(args.test_pdf)
                print(f"Successfully extracted {len(images)} pages")
                print("PDF processing test passed!")
            except Exception as e:
                print(f"PDF processing test failed: {e}")
        else:
            print(f"PDF file not found: {args.test_pdf}")
    
    else:
        test_pdf_processing()

if __name__ == "__main__":
    main()