import torch
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

from networks.models import Colorizer
from denoising.denoiser import FFDNetDenoiser
from utils.utils import resize_pad

class MangaColorizator:
    def __init__(self, device, generator_path = 'networks/generator.zip', extractor_path = 'networks/extractor.pth'):
        self.colorizer = Colorizer().to(device)
        self.colorizer.generator.load_state_dict(torch.load(generator_path, map_location = device))
        self.colorizer = self.colorizer.eval()
        
        self.denoiser = FFDNetDenoiser(device)
        
        self.current_image = None
        self.current_hint = None
        self.current_pad = None
        
        # Batch processing attributes
        self.current_images = None
        self.current_hints = None
        self.current_pads = None
        
        self.device = device
        
    def set_image(self, image, size = 576, apply_denoise = True, denoise_sigma = 25, transform = ToTensor()):
        if (size % 32 != 0):
            raise RuntimeError("size is not divisible by 32")
        
        if apply_denoise:
            image = self.denoiser.get_denoised_image(image, sigma = denoise_sigma)
        
        image, self.current_pad = resize_pad(image, size)
        self.current_image = transform(image).unsqueeze(0).to(self.device)
        self.current_hint = torch.zeros(1, 4, self.current_image.shape[2], self.current_image.shape[3]).float().to(self.device)
    
    def update_hint(self, hint, mask):
        '''
        Args:
           hint: numpy.ndarray with shape (self.current_image.shape[2], self.current_image.shape[3], 3)
           mask: numpy.ndarray with shape (self.current_image.shape[2], self.current_image.shape[3])
        '''
        
        if issubclass(hint.dtype.type, np.integer):
            hint = hint.astype('float32') / 255
            
        hint = (hint - 0.5) / 0.5
        hint = torch.FloatTensor(hint).permute(2, 0, 1)
        mask = torch.FloatTensor(np.expand_dims(mask, 0))

        self.current_hint = torch.cat([hint * mask, mask], 0).unsqueeze(0).to(self.device)

    def colorize(self):
        with torch.no_grad():
            fake_color, _ = self.colorizer(torch.cat([self.current_image, self.current_hint], 1))
            fake_color = fake_color.detach()

        result = fake_color[0].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5

        if self.current_pad[0] != 0:
            result = result[:-self.current_pad[0]]
        if self.current_pad[1] != 0:
            result = result[:, :-self.current_pad[1]]
            
        return result.numpy()
    
    def set_images(self, images, size=576, apply_denoise=True, denoise_sigma=25, transform=ToTensor()):
        """
        Set multiple images for batch processing
        Args:
            images: List of numpy arrays or single numpy array
            size: Target size for processing
            apply_denoise: Whether to apply denoising
            denoise_sigma: Denoising strength
            transform: Transform to apply to images
        """
        if (size % 32 != 0):
            raise RuntimeError("size is not divisible by 32")
        
        if not isinstance(images, list):
            images = [images]
        
        processed_images = []
        pads = []
        
        for image in images:
            if apply_denoise:
                image = self.denoiser.get_denoised_image(image, sigma=denoise_sigma)
            
            image, pad = resize_pad(image, size)
            processed_images.append(transform(image))
            pads.append(pad)
        
        # Stack images into batch tensor
        self.current_images = torch.stack(processed_images).to(self.device)
        self.current_pads = pads
        
        # Initialize hints for each image in batch
        batch_size = self.current_images.shape[0]
        self.current_hints = torch.zeros(batch_size, 4, self.current_images.shape[2], self.current_images.shape[3]).float().to(self.device)
    
    def update_hints(self, hints, masks):
        """
        Update hints for batch processing
        Args:
           hints: List of numpy arrays with shape (H, W, 3) for each image
           masks: List of numpy arrays with shape (H, W) for each image
        """
        if not isinstance(hints, list):
            hints = [hints]
        if not isinstance(masks, list):
            masks = [masks]
            
        batch_hints = []
        
        for hint, mask in zip(hints, masks):
            if issubclass(hint.dtype.type, np.integer):
                hint = hint.astype('float32') / 255
                
            hint = (hint - 0.5) / 0.5
            hint = torch.FloatTensor(hint).permute(2, 0, 1)
            mask = torch.FloatTensor(np.expand_dims(mask, 0))
            
            batch_hints.append(torch.cat([hint * mask, mask], 0))
        
        self.current_hints = torch.stack(batch_hints).to(self.device)
    
    def colorize_batch(self):
        """
        Colorize a batch of images
        Returns:
            List of colorized images as numpy arrays
        """
        with torch.no_grad():
            fake_colors, _ = self.colorizer(torch.cat([self.current_images, self.current_hints], 1))
            fake_colors = fake_colors.detach()

        results = []
        for i in range(fake_colors.shape[0]):
            result = fake_colors[i].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5
            
            # Apply padding removal for this specific image
            pad = self.current_pads[i]
            if pad[0] != 0:
                result = result[:-pad[0]]
            if pad[1] != 0:
                result = result[:, :-pad[1]]
                
            results.append(result.numpy())
            
        return results
    
    def process_images_batch(self, images, size=576, apply_denoise=True, denoise_sigma=25):
        """
        Complete batch processing pipeline
        Args:
            images: List of image paths or numpy arrays
            size: Target size for processing
            apply_denoise: Whether to apply denoising
            denoise_sigma: Denoising strength
        Returns:
            List of colorized images as numpy arrays
        """
        # Load images if paths are provided
        if isinstance(images[0], str):
            loaded_images = []
            for img_path in images:
                loaded_images.append(plt.imread(img_path))
            images = loaded_images
        
        self.set_images(images, size, apply_denoise, denoise_sigma)
        return self.colorize_batch()
