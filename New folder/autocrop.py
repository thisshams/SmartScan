from PIL import Image
from skimage.io import imread
from skimage.morphology import convex_hull_image
from skimage.color import rgb2gray
im = imread('L_2d.jpg')
plt.imshow(im)
plt.title('input image')
plt.show()
# create a binary image
im1 = 1 - rgb2gray(im)
threshold = 0.5
im1[im1 <= threshold] = 0
im1[im1 > threshold] = 1
chull = convex_hull_image(im1)
plt.imshow(chull)
plt.title('convex hull in the binary image')
plt.show()
imageBox = Image.fromarray((chull*255).astype(np.uint8)).getbbox()
cropped = Image.fromarray(im).crop(imageBox)
cropped.save('L_2d_cropped.jpg')
plt.imshow(cropped)
plt.show()
