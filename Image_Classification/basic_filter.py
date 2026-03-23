from PIL import Image, ImageOps, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import os

def apply_nat20_filter(image_path, output_path="nat20_map.png"):
    try:
        # 1. Open the original image (No resizing so it stays high-def!)
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        # 2. Apply the "Old Parchment" Sepia Tone
        gray = img.convert("L") # Turn it black and white first
        sepia = ImageOps.colorize(gray, black="#302013", white="#e6d5b8") # Add parchment colors

        # 3. Create the "Vignette" (Darkened edges like an old map)
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, width, height), fill=255)
        # Blur the circle heavily so it fades smoothly into the corners
        mask = mask.filter(ImageFilter.GaussianBlur(radius=min(width, height) // 4))
        
        black = Image.new('RGB', (width, height), (0, 0, 0))
        final_img = Image.composite(sepia, black, mask)

        # 4. Save the Final Masterpiece
        plt.imshow(final_img)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"\nSUCCESS: Nat 20 map saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    print("Nat 20 D&D Map Filter (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
        
       
        base, ext = os.path.splitext(image_path)
        output_file = f"{base}_nat20{ext}"
        apply_nat20_filter(image_path, output_file)