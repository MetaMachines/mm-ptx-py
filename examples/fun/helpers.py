import torch

def render_video(
    path,
    tensor,
    fps = 60
):
    import cv2

    (num_images, height, width, num_channels) = tensor.shape
    assert(num_channels == 4)

    video_path = path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or 'XVID' for AVI
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(num_images):

        rgba_tensor = tensor[i].view(torch.uint8).reshape(height, width, 4)

        frame_np = rgba_tensor.cpu().numpy()
        bgr_np = frame_np[..., :3]
        writer.write(bgr_np)
    writer.release()
    print(f"Video saved to {video_path}")

def generate_gif(
    path,
    tensor,
    duration=100
):
    from PIL import Image

    frames = []
    for i in range(tensor.shape[0]):
        frame_np = tensor[i].cpu().numpy()  # [H, W, 4] BGRA uint8
        # Swap B and R channels for RGBA (PIL expects RGBA)
        rgba_np = frame_np[..., [2, 1, 0, 3]]
        img = Image.fromarray(rgba_np, mode='RGBA')
        frames.append(img)

    # Save as GIF: First frame saves, append others
    # Adjust duration (ms per frame) and loop (0 for infinite)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0)

def show_image(
    path,
    tensor,
    show = True,
    image_num = 0
):
    (num_images, height, width, num_channels) = tensor.shape

    rgba_tensor = tensor[image_num].view(torch.uint8).reshape(height, width, 4)

    # To fully "read out" or visualize:
    # Bring to CPU and convert to NumPy for easy access/manipulation
    rgba_array = rgba_tensor.cpu().numpy()  # Shape: [H, W, 4], dtype=uint8

    # Optional: Convert to PIL Image for saving/displaying as RGBA PNG
    from PIL import Image
    pil_image = Image.fromarray(rgba_array, mode='RGBA')
    pil_image.save(path)  # Save as PNG (supports transparency)
    pil_image.show()  # Display if in a GUI environment

def montage_tensors(
    tensors_list,
    dim_x, dim_y
):
    num_images, height, width, channels = tensors_list[0].shape
    assert(channels == 4)

    tensors = torch.stack(tensors_list).view(dim_x, dim_y, *tensors_list[0].shape)
    tensors = tensors.permute(2, 0, 3, 1, 4, 5)

    tensors = tensors.reshape(num_images, dim_y, height, dim_x * width, channels)
    tensors = tensors.reshape(num_images, dim_y * height, dim_x * width, channels)
    tensors = tensors.contiguous()
    return tensors
    
