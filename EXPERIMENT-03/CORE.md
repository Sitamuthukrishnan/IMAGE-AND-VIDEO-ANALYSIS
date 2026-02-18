#CORE:

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load Image
# ----------------------------
img = Image.open("sita.jpg")
img_np = np.array(img)

plt.imshow(img)
plt.axis("off")
plt.title("Original Image")
plt.show()


# ----------------------------
# Helper: Nearest Neighbor
# ----------------------------
def sample_nearest(image, x, y):
    h, w = image.shape[:2]
    x = int(round(x))
    y = int(round(y))
    if 0 <= x < w and 0 <= y < h:
        return image[y, x]
    return np.array([0, 0, 0])


# ----------------------------
# Rotate Image
# ----------------------------
def rotate_image(image, angle):
    angle = np.deg2rad(angle)
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    output = np.zeros_like(image)

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    for y in range(h):
        for x in range(w):
            tx = x - cx
            ty = y - cy
            src_x = cos_a * tx + sin_a * ty + cx
            src_y = -sin_a * tx + cos_a * ty + cy
            output[y, x] = sample_nearest(image, src_x, src_y)

    return output


# ----------------------------
# Scale Image
# ----------------------------
def scale_image(image, sx, sy):
    h, w = image.shape[:2]
    new_h = int(h * sy)
    new_w = int(w * sx)
    output = np.zeros((new_h, new_w, 3), dtype=image.dtype)

    for y in range(new_h):
        for x in range(new_w):
            src_x = x / sx
            src_y = y / sy
            output[y, x] = sample_nearest(image, src_x, src_y)

    return output


# ----------------------------
# Skew (Shear)
# ----------------------------
def skew_image(image, skew_x, skew_y):
    h, w = image.shape[:2]
    output = np.zeros_like(image)

    for y in range(h):
        for x in range(w):
            src_x = x - skew_x * y
            src_y = y - skew_y * x
            output[y, x] = sample_nearest(image, src_x, src_y)

    return output


# ----------------------------
# Affine Transform
# ----------------------------
def affine_transform(image, src_pts, dst_pts):
    src = np.hstack([src_pts, np.ones((3, 1))])
    matrix = np.linalg.lstsq(src, dst_pts, rcond=None)[0].T

    h, w = image.shape[:2]
    output = np.zeros_like(image)

    inv_matrix = np.linalg.inv(
        np.vstack([matrix, [0, 0, 1]])
    )

    for y in range(h):
        for x in range(w):
            sx, sy, _ = inv_matrix @ np.array([x, y, 1])
            output[y, x] = sample_nearest(image, sx, sy)

    return output


# ----------------------------
# Perspective Transform
# ----------------------------
def perspective_transform(image, src_pts, dst_pts):
    A, B = [], []

    for (x, y), (u, v) in zip(src_pts, dst_pts):
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
        B.append(u)
        B.append(v)

    A = np.array(A)
    B = np.array(B)

    H = np.append(np.linalg.solve(A, B), 1).reshape(3, 3)
    inv_H = np.linalg.inv(H)

    h, w = image.shape[:2]
    output = np.zeros_like(image)

    for y in range(h):
        for x in range(w):
            px, py, pz = inv_H @ np.array([x, y, 1])
            px /= pz
            py /= pz
            output[y, x] = sample_nearest(image, px, py)

    return output


# ----------------------------
# Apply & Display Results
# ----------------------------

rotated = rotate_image(img_np, 45)
plt.imshow(rotated)
plt.axis("off")
plt.title("Rotated Image")
plt.show()

scaled = scale_image(img_np, 1.5, 1.5)
plt.imshow(scaled)
plt.axis("off")
plt.title("Scaled Image")
plt.show()

skewed = skew_image(img_np, 0.3, 0.0)
plt.imshow(skewed)
plt.axis("off")
plt.title("Skewed Image")
plt.show()

src_affine = np.float32([[50, 50], [200, 50], [50, 200]])
dst_affine = np.float32([[10, 100], [200, 50], [100, 250]])
affine_img = affine_transform(img_np, src_affine, dst_affine)
plt.imshow(affine_img)
plt.axis("off")
plt.title("Affine Transform")
plt.show()

src_persp = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
dst_persp = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
persp_img = perspective_transform(img_np, src_persp, dst_persp)
plt.imshow(persp_img)
plt.axis("off")
plt.title("Perspective Transform")
plt.show()

#Output:
<img width="466" height="248" alt="Screenshot 2026-02-18 093300" src="https://github.com/user-attachments/assets/0e2390a6-386b-4c00-b102-ba2c96cdf6e5" />
<img width="472" height="248" alt="Screenshot 2026-02-18 093346" src="https://github.com/user-attachments/assets/f21d96da-f6c8-41c2-b764-4edee89f1732" />
<img width="474" height="257" alt="Screenshot 2026-02-18 093426" src="https://github.com/user-attachments/assets/f66b073d-aa1e-4890-abd9-9a99c0f2a69e" />
<img width="604" height="300" alt="Screenshot 2026-02-18 094542" src="https://github.com/user-attachments/assets/da4735dc-32a6-44d5-b15a-c52e289b313f" />
<img width="651" height="336" alt="Screenshot 2026-02-18 094548" src="https://github.com/user-attachments/assets/5d91e83b-c6d2-403e-b1c1-5830dd8e4d43" />
<img width="605" height="275" alt="Screenshot 2026-02-18 094557" src="https://github.com/user-attachments/assets/6a29ba4a-06eb-463f-844a-470b20db0a5d" />





