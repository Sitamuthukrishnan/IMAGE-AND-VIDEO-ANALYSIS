#TESTCASE:SCALING ZOOMIN AND ZOOMOUT

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load & Display Image
# ----------------------------
img = Image.open("sita.jpg")
img_np = np.array(img)

plt.imshow(img)
plt.axis("off")
plt.title("Original Image")
plt.show()


# ----------------------------
# Nearest Neighbor Sampling
# ----------------------------
def sample_nearest(image, x, y):
    h, w = image.shape[:2]
    x = int(round(x))
    y = int(round(y))
    if 0 <= x < w and 0 <= y < h:
        return image[y, x]
    return np.array([0, 0, 0])


# ----------------------------
# Scaling Function
# ----------------------------
def scale_image(image, scale_x, scale_y):
    h, w = image.shape[:2]
    new_h = int(h * scale_y)
    new_w = int(w * scale_x)

    output = np.zeros((new_h, new_w, 3), dtype=image.dtype)

    for y in range(new_h):
        for x in range(new_w):
            src_x = x / scale_x
            src_y = y / scale_y
            output[y, x] = sample_nearest(image, src_x, src_y)

    return output


# ----------------------------
# Zoom In (center crop + scale)
# ----------------------------
def zoom_in(image, zoom_factor):
    h, w = image.shape[:2]

    crop_h = int(h / zoom_factor)
    crop_w = int(w / zoom_factor)

    start_y = (h - crop_h) // 2
    start_x = (w - crop_w) // 2

    cropped = image[start_y:start_y + crop_h, start_x:start_x + crop_w]

    return scale_image(cropped, zoom_factor, zoom_factor)


# ----------------------------
# Zoom Out (shrink + pad)
# ----------------------------
def zoom_out(image, zoom_factor):
    h, w = image.shape[:2]

    scaled = scale_image(image, 1 / zoom_factor, 1 / zoom_factor)

    new_h, new_w = scaled.shape[:2]
    output = np.zeros_like(image)

    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2

    output[start_y:start_y + new_h, start_x:start_x + new_w] = scaled

    return output


# ----------------------------
# APPLY & DISPLAY
# ----------------------------

# Scaling
scaled_img = scale_image(img_np, 1.5, 1.5)
plt.imshow(scaled_img)
plt.axis("off")
plt.title("Scaled Image (1.5x)")
plt.show()

# Zoom In
zoomed_in = zoom_in(img_np, 2)
plt.imshow(zoomed_in)
plt.axis("off")
plt.title("Zoom In (2x)")
plt.show()

# Zoom Out
zoomed_out = zoom_out(img_np, 2)
plt.imshow(zoomed_out)
plt.axis("off")
plt.title("Zoom Out (2x)")
plt.show()

#Output:
<img width="214" height="432" alt="Screenshot 2026-02-18 093524" src="https://github.com/user-attachments/assets/b957332e-07b3-414b-96e2-8d40e58298b3" />
