#!/usr/bin/env python3
"""
Example script demonstrating batch processing capabilities
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from colorizator import MangaColorizator

def test_batch_processing():
    """
    Test the batch processing functionality
    """
    # Initialize colorizer
    device = 'cpu'  # Change to 'cuda' if GPU available
    generator_path = 'networks/generator.zip'
    extractor_path = 'networks/extractor.pth'
    
    print("Initializing MangaColorizator...")
    colorizer = MangaColorizator(device, generator_path, extractor_path)
    
    # Example: Create dummy images for testing (replace with real image paths)
    # For real usage, provide a list of image file paths or loaded images
    print("Creating dummy test images...")
    dummy_images = []
    for i in range(3):
        # Create a dummy grayscale image (you would load real images here)
        dummy_img = np.random.rand(256, 256, 3).astype(np.float32)
        dummy_images.append(dummy_img)
    
    # Process batch
    print("Processing batch of images...")
    try:
        colorized_results = colorizer.process_images_batch(
            dummy_images,
            size=576,  # Must be divisible by 32
            apply_denoise=True,
            denoise_sigma=25
        )
        
        print(f"Successfully processed {len(colorized_results)} images in batch")
        
        # Save results (optional)
        for i, result in enumerate(colorized_results):
            output_path = f"batch_output_{i}.png"
            plt.imsave(output_path, result)
            print(f"Saved: {output_path}")
            
    except Exception as e:
        print(f"Error during batch processing: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing batch processing functionality...")
    success = test_batch_processing()
    
    if success:
        print("Batch processing test completed successfully!")
        print("\nUsage examples:")
        print("1. Single image: python inference.py -p image.jpg")
        print("2. Directory with batch size 4: python inference.py -p /path/to/images/ -b 4")
        print("3. GPU batch processing: python inference.py -p /path/to/images/ -b 8 -g")
    else:
        print("Batch processing test failed.")