CODE:


w, h = img.size


half_w = w // 2
half_h = h // 2

img1 = img.crop((0, 0, half_w, half_h))        
img2 = img.crop((half_w, 0, w, half_h))        
img3 = img.crop((0, half_h, half_w, h))      
img4 = img.crop((half_w, half_h, w, h))        


fig, axs = plt.subplots(2, 2, figsize=(6, 6))

axs[0, 0].imshow(img1)
axs[0, 0].set_title("Quadrant 1")

axs[0, 1].imshow(img2)
axs[0, 1].set_title("Quadrant 2")

axs[1, 0].imshow(img3)
axs[1, 0].set_title("Quadrant 3")

axs[1, 1].imshow(img4)
axs[1, 1].set_title("Quadrant 4")

for ax in axs.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()

OUTPUT:
<img width="649" height="489" alt="Screenshot 2026-02-18 092958" src="https://github.com/user-attachments/assets/31297da8-3285-43f7-bcdc-93df2ee20f7f" />
