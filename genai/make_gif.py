import os
import glob
from PIL import Image

def create_gif(image_folder, output_path="gan_progress.gif", duration=200):
    # Get all image paths sorted by epoch
    images = sorted(glob.glob(os.path.join(image_folder, "generated_img_*.png")))

    if not images:
        print("No images found. Make sure images are saved in:", image_folder)
        return

    frames = [Image.open(img_path) for img_path in images]

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,  # time per frame in ms
        loop=0
    )
    print(f"GIF saved as: {output_path}")

if __name__ == "__main__":
    create_gif(image_folder="generated_images", output_path="gan_training.gif")
