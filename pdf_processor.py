import fitz  # PyMuPDF
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class PDFProcessor:
    def __init__(self, dpi=150):
        """
        Initialize PDF processor
        Args:
            dpi: Resolution for PDF to image conversion (higher = better quality, more memory)
        """
        self.dpi = dpi
    
    def pdf_to_images(self, pdf_path, output_dir=None):
        """
        Convert PDF pages to images
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images (optional)
        Returns:
            List of numpy arrays (images) and list of page info
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        images = []
        page_info = []
        
        print(f"Processing PDF: {pdf_path}")
        print(f"Total pages: {len(pdf_document)}")
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Convert page to image
            mat = fitz.Matrix(self.dpi/72, self.dpi/72)  # 72 is default DPI
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            pil_img = Image.open(fitz.io.BytesIO(img_data))
            
            # Convert to numpy array
            img_array = np.array(pil_img).astype(np.float32) / 255.0
            
            images.append(img_array)
            page_info.append({
                'page_num': page_num + 1,
                'width': pix.width,
                'height': pix.height
            })
            
            # Optionally save individual page images
            if output_dir:
                page_filename = f"page_{page_num + 1:03d}.png"
                page_path = os.path.join(output_dir, page_filename)
                plt.imsave(page_path, img_array)
                print(f"Saved page {page_num + 1} to {page_path}")
        
        pdf_document.close()
        print(f"Extracted {len(images)} pages from PDF")
        
        return images, page_info
    
    def save_colorized_pdf_pages(self, colorized_images, page_info, output_dir, pdf_name):
        """
        Save colorized images as individual pages
        Args:
            colorized_images: List of colorized image arrays
            page_info: List of page information
            output_dir: Output directory
            pdf_name: Original PDF name (for naming)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        for i, (img, info) in enumerate(zip(colorized_images, page_info)):
            filename = f"{pdf_name}_colorized_page_{info['page_num']:03d}.png"
            filepath = os.path.join(output_dir, filename)
            
            plt.imsave(filepath, img)
            saved_files.append(filepath)
            print(f"Saved colorized page {info['page_num']} to {filepath}")
        
        return saved_files
    
    def create_colorized_pdf(self, colorized_images, page_info, output_pdf_path, original_pdf_path=None):
        """
        Create a new PDF from colorized images
        Args:
            colorized_images: List of colorized image arrays
            page_info: List of page information
            output_pdf_path: Path for output PDF
            original_pdf_path: Original PDF path for reference
        """
        try:
            # Create new PDF document
            new_pdf = fitz.open()
            
            for i, (img, info) in enumerate(zip(colorized_images, page_info)):
                # Convert numpy array back to PIL Image
                img_uint8 = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_uint8)
                
                # Convert PIL image to bytes
                img_bytes = fitz.io.BytesIO()
                pil_img.save(img_bytes, format='PNG')
                img_bytes = img_bytes.getvalue()
                
                # Create new page with same dimensions as original
                page_width = info['width']
                page_height = info['height'] 
                
                # Add page to PDF
                page = new_pdf.new_page(width=page_width, height=page_height)
                
                # Insert image
                img_rect = fitz.Rect(0, 0, page_width, page_height)
                page.insert_image(img_rect, stream=img_bytes)
            
            # Save the new PDF
            new_pdf.save(output_pdf_path)
            new_pdf.close()
            
            print(f"Created colorized PDF: {output_pdf_path}")
            return output_pdf_path
            
        except Exception as e:
            print(f"Error creating PDF: {e}")
            return None

def test_pdf_processor():
    """
    Test function for PDF processor
    """
    processor = PDFProcessor(dpi=150)
    
    # This is just a test - replace with actual PDF path
    test_pdf = "test.pdf"
    
    if os.path.exists(test_pdf):
        try:
            images, page_info = processor.pdf_to_images(test_pdf, "extracted_pages")
            print(f"Successfully extracted {len(images)} pages")
            return True
        except Exception as e:
            print(f"Error testing PDF processor: {e}")
            return False
    else:
        print("No test PDF found. PDF processor module ready for use.")
        return True

if __name__ == "__main__":
    test_pdf_processor()