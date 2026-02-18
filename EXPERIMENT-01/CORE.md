CORE:
# T-Pyramid of an image

# PROGRAM:
import cv2
import numpy as np

def build_t_pyramid(image, levels):
    pyramid = [image]

    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid


def main():
    image_path = "img.jpg"
    levels = 3
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Error: would not load the image")
        return

    t_pyramid = build_t_pyramid(original_image, levels)

    for i, level_image in enumerate(t_pyramid):
        cv2.imshow(f"Levels {i}", level_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

OUTPUT:
<img width="333" height="275" alt="image" src="https://github.com/user-attachments/assets/4ffd2a75-6563-4750-96e7-4b5161cf6488" />
