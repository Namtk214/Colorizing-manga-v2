import os
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from colorizator import MangaColorizator
from pdf_processor import PDFProcessor

def process_image(image, colorizator, args):
    colorizator.set_image(image, args.size, args.denoiser, args.denoiser_sigma)
        
    return colorizator.colorize()
    
def colorize_single_image(image_path, save_path, colorizator, args):
    
        image = plt.imread(image_path)

        colorization = process_image(image, colorizator, args)
        
        plt.imsave(save_path, colorization)
        
        return True
    

def colorize_images(target_path, colorizator, args):
    images = os.listdir(args.path)
    
    # Filter valid image files
    valid_images = []
    for image_name in images:
        file_path = os.path.join(args.path, image_name)
        
        if os.path.isdir(file_path):
            continue
        
        name, ext = os.path.splitext(image_name)
        if ext.lower() in ('.jpg', '.png', '.jpeg'):
            valid_images.append((file_path, image_name))
    
    if getattr(args, 'batch_size', 1) > 1 and len(valid_images) > 1:
        colorize_images_batch(target_path, valid_images, colorizator, args)
    else:
        # Single image processing (original method)
        for file_path, image_name in valid_images:
            name, ext = os.path.splitext(image_name)
            if (ext != '.png'):
                image_name = name + '.png'
            
            print(file_path)
            
            save_path = os.path.join(target_path, image_name)
            colorize_single_image(file_path, save_path, colorizator, args)

def colorize_images_batch(target_path, valid_images, colorizator, args):
    """
    Process images in batches for improved performance
    """
    batch_size = getattr(args, 'batch_size', 4)
    
    for i in range(0, len(valid_images), batch_size):
        batch = valid_images[i:i + batch_size]
        
        # Extract file paths and names
        file_paths = [item[0] for item in batch]
        image_names = [item[1] for item in batch]
        
        print(f"Processing batch {i//batch_size + 1}: {[os.path.basename(fp) for fp in file_paths]}")
        
        try:
            # Process batch
            colorized_images = colorizator.process_images_batch(
                file_paths, 
                args.size, 
                args.denoiser, 
                args.denoiser_sigma
            )
            
            # Save results
            for j, (colorized_img, image_name) in enumerate(zip(colorized_images, image_names)):
                name, ext = os.path.splitext(image_name)
                if ext != '.png':
                    image_name = name + '.png'
                
                save_path = os.path.join(target_path, image_name)
                plt.imsave(save_path, colorized_img)
                
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Fallback to single image processing for this batch
            print("Falling back to single image processing...")
            for file_path, image_name in batch:
                name, ext = os.path.splitext(image_name)
                if ext != '.png':
                    image_name = name + '.png'
                
                save_path = os.path.join(target_path, image_name)
                try:
                    colorize_single_image(file_path, save_path, colorizator, args)
                except Exception as single_e:
                    print(f"Error processing {file_path}: {single_e}")

def colorize_pdf(pdf_path, colorizator, args):
    """
    Process a PDF file by converting pages to images and colorizing them
    """
    print(f"Processing PDF: {pdf_path}")
    
    # Initialize PDF processor
    pdf_processor = PDFProcessor(dpi=getattr(args, 'pdf_dpi', 150))
    
    # Extract PDF name for output naming
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(os.path.dirname(pdf_path), f"{pdf_name}_colorized")
    
    try:
        # Convert PDF to images
        print("Converting PDF pages to images...")
        images, page_info = pdf_processor.pdf_to_images(pdf_path)
        
        print(f"Extracted {len(images)} pages from PDF")
        
        # Process images with batch processing
        batch_size = getattr(args, 'batch_size', 4)
        print(f"Colorizing pages with batch size: {batch_size}")
        
        colorized_images = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_pages = list(range(i + 1, min(i + batch_size + 1, len(images) + 1)))
            
            print(f"Processing pages {batch_pages[0]}-{batch_pages[-1]}...")
            
            try:
                # Use batch processing
                batch_results = colorizator.process_images_batch(
                    batch,
                    args.size,
                    args.denoiser,
                    args.denoiser_sigma
                )
                colorized_images.extend(batch_results)
                
            except Exception as e:
                print(f"Batch processing failed for pages {batch_pages}: {e}")
                print("Falling back to single image processing...")
                
                # Fallback to single image processing
                for j, img in enumerate(batch):
                    try:
                        colorizator.set_image(img, args.size, args.denoiser, args.denoiser_sigma)
                        result = colorizator.colorize()
                        colorized_images.append(result)
                        print(f"Processed page {i + j + 1}")
                    except Exception as single_e:
                        print(f"Error processing page {i + j + 1}: {single_e}")
                        # Add original image as fallback
                        colorized_images.append(img)
        
        # Save results
        print("Saving colorized pages...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual page images
        saved_files = pdf_processor.save_colorized_pdf_pages(
            colorized_images, page_info, output_dir, pdf_name
        )
        
        # Optionally create a new colorized PDF
        if getattr(args, 'create_pdf', True):
            output_pdf_path = os.path.join(output_dir, f"{pdf_name}_colorized.pdf")
            pdf_processor.create_colorized_pdf(
                colorized_images, page_info, output_pdf_path, pdf_path
            )
        
        print(f"Processing complete! Results saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return False
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-gen", "--generator", default = 'networks/generator.zip')
    parser.add_argument("-ext", "--extractor", default = 'networks/extractor.pth')
    parser.add_argument('-g', '--gpu', dest = 'gpu', action = 'store_true')
    parser.add_argument('-nd', '--no_denoise', dest = 'denoiser', action = 'store_false')
    parser.add_argument("-ds", "--denoiser_sigma", type = int, default = 25)
    parser.add_argument("-s", "--size", type = int, default = 576)
    parser.add_argument("-b", "--batch_size", type = int, default = 1, help = "Number of images to process in each batch (default: 1 for single image processing)")
    parser.add_argument("--pdf_dpi", type = int, default = 150, help = "DPI for PDF to image conversion (default: 150)")
    parser.add_argument("--no_pdf_output", dest = "create_pdf", action = "store_false", help = "Don't create output PDF, only save individual images")
    parser.set_defaults(gpu = False)
    parser.set_defaults(denoiser = True)
    parser.set_defaults(create_pdf = True)
    args = parser.parse_args()
    
    return args

    
if __name__ == "__main__":
    
    args = parse_args()
    
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
        
    colorizer = MangaColorizator(device, args.generator, args.extractor)
    
    if os.path.isdir(args.path):
        colorization_path = os.path.join(args.path, 'colorization')
        if not os.path.exists(colorization_path):
            os.makedirs(colorization_path)
              
        colorize_images(colorization_path, colorizer, args)
        
    elif os.path.isfile(args.path):
        
        split = os.path.splitext(args.path)
        
        if split[1].lower() == '.pdf':
            # Process PDF file
            print("Detected PDF file, processing...")
            colorize_pdf(args.path, colorizer, args)
            
        elif split[1].lower() in ('.jpg', '.png', '.jpeg'):
            # Process single image
            new_image_path = split[0] + '_colorized' + '.png'
            colorize_single_image(args.path, new_image_path, colorizer, args)
        else:
            print('Unsupported format. Supported formats: .jpg, .png, .jpeg, .pdf')
    else:
        print('Wrong path')
    
