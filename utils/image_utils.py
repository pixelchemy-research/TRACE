import torchvision
import PIL
from math import ceil
import torchvision.transforms.functional as F


def resize_image(image, size=768):
    tensor_image = F.to_tensor(image)
    resized_image = F.resize(tensor_image, size, antialias=True)
    return resized_image


def save_images(images, save_path, rows=1, cols=None):
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    elif images.size(1) > 3:
        images = images[:, :3]

    if cols is None:
        cols = images.size(0) // rows

    _, _, h, w = images.shape
    grid = PIL.Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(images):
        img = torchvision.transforms.functional.to_pil_image(img.clamp(0, 1).cpu())
        grid.paste(img, box=(i % cols * w, i // cols * h))

    grid.save(save_path)


def calculate_latent_sizes(height=1024, width=1024, batch_size=4, compression_factor_b=42.67, compression_factor_a=4.0):
    resolution_multiple = 42.67
    latent_height = ceil(height / compression_factor_b)
    latent_width = ceil(width / compression_factor_b)
    stage_c_latent_shape = (batch_size, 16, latent_height, latent_width)
    
    latent_height = ceil(height / compression_factor_a)
    latent_width = ceil(width / compression_factor_a)
    stage_b_latent_shape = (batch_size, 4, latent_height, latent_width)
    
    return stage_c_latent_shape, stage_b_latent_shape