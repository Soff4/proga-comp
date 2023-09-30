import numpy as np
import cv2

image_01 = cv2.imread('capybara.jpeg', cv2.IMREAD_COLOR)
image_02 = cv2.imread('shrek.jpeg', cv2.IMREAD_COLOR)
image_03 = cv2.imread('text_01.jpeg', cv2.IMREAD_COLOR)
image_04 = cv2.imread('text_02.jpeg', cv2.IMREAD_COLOR)

def handlerGrayscale(img):
    height, width, _ = img.shape
    gray_image = np.zeros((height, width), dtype=np.uint8)
    for h in range(height):
        for w in range(width):
            blue, green, red = img[h, w]
            gray_value = (int(red) + int(green) + int(blue)) // 3
            gray_image[h, w] = gray_value
    return gray_image

def handlerBinaryzation(img):
    window_size = 15
    size = window_size // 2

    local_mean = np.zeros_like(img, dtype=np.float32)
    for i in range(size, img.shape[0] - size):
        for j in range(size, img.shape[1] - size):
            window = img[i - size:i + size + 1, j - size:j + size + 1]
            local_mean[i, j] = np.mean(window)

    local_std = np.zeros_like(img, dtype=np.float32)
    for i in range(size, img.shape[0] - size):
        for j in range(size, img.shape[1] - size):
            window = img[i - size:i + size + 1, j - size:j + size + 1]
            local_std[i, j] = np.std(window)

    k = -0.2
    binary_image = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            threshold = local_mean[i, j] + k * local_std[i, j]
            if img[i, j] > threshold:
                binary_image[i, j] = 255
            else:
                binary_image[i, j] = 0

    return binary_image

def handlerRemoveBg(img):
    threshold = 128
    mask = img > threshold
    remove_image = img.copy()
    remove_image[~mask] = 0
    return remove_image

capy_gray_image = handlerGrayscale(image_01)
capy_binary_image = handlerBinaryzation(capy_gray_image)
capy_remove_image = handlerRemoveBg(image_01)
cv2.imwrite('capy_binary_image.jpeg', capy_binary_image)
cv2.imwrite('capy_remove_image.jpeg', capy_remove_image)

shrek_gray_image = handlerGrayscale(image_02)
shrek_binary_image = handlerBinaryzation(shrek_gray_image)
shrek_remove_image = handlerRemoveBg(image_02)
cv2.imwrite('shrek_binary_image.jpeg', shrek_binary_image)
cv2.imwrite('shrek_remove_image.jpeg', shrek_remove_image)

text_01_gray_image = handlerGrayscale(image_03)
text_01_binary_image = handlerBinaryzation(text_01_gray_image)
text_01_remove_image = handlerRemoveBg(image_03)
cv2.imwrite('text_01_binary_image.jpeg', text_01_binary_image)
cv2.imwrite('text_01_remove_image.jpeg', text_01_remove_image)

text_02_gray_image = handlerGrayscale(image_04)
text_02_binary_image = handlerBinaryzation(text_02_gray_image)
text_02_remove_image = handlerRemoveBg(image_04)
cv2.imwrite('text_02_binary_image.jpeg', text_02_binary_image)
cv2.imwrite('text_02_remove_image.jpeg', text_02_remove_image)


