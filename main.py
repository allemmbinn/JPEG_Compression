import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import itertools

"""
Function : Convert RGB Image to YCbCr Image
Input    : RGB Image
Output   : YCbCr Image
"""
def rgb_to_ycbcr(image):
    ycbcr_image = np.zeros(image.shape)
    # Transformation Matrix
    T = np.array([
                [ 0.299, 0.587, 0.114],
                [-0.169,-0.331,-0.500],
                [ 0.500,-0.419,-0.081]])
    # Conversion
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rgb_pixel = image[i, j, :]
            ycbcr_image[i, j, :] = T@rgb_pixel + np.array([0, 128, 128])
    return ycbcr_image

"""
Function : Convert YCbCr Image to RGB Image
Input    : YCbCr Image
Output   : RGB Image
"""
def ycbcr_to_rgb(image):
    rgb_image = np.zeros(image.shape)
    # Transformation Matrix
    T = np.array([
                [1.0, 0.000, 1.402],
                [1.0,-0.344,-0.714],
                [1.0, 1.772, 0.000]])
    # Conversion
    for i in range(image.shape):
        for j in range(image.shape):
            ycbcr_to_rgb = image[i, j, :]
            rgb_image[i, j, :] = T@(ycbcr_to_rgb - np.array([0, 128, 128]))
    return rgb_image
    
"""
Function : Subsample Cb and Cr
TODO : Look into this function
"""
# Sub-sampling (4:2:0) 
def subsample(ycbcr_image):
    Y, Cb, Cr = ycbcr_image[:,:,0], ycbcr_image[:,:,1], ycbcr_image[:,:,2]
    Cb = Cb[::2, ::2]
    Cr = Cr[::2, ::2]
    return Y, Cb, Cr

# TODO : Change this
def upsample(Y, Cb, Cr):
    Cb_up = np.repeat(np.repeat(Cb, 2, axis=0), 2, axis=1)
    Cr_up = np.repeat(np.repeat(Cr, 2, axis=0), 2, axis=1)
    return np.dstack((Y, Cb_up, Cr_up))

# Discrete Cosine Transform (DCT) and Inverse DCT
def apply_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

"""
Function : DCT Transform on Spatial Image
Input    : YCbCr Image
Output   : RGB Image
"""
def dct_transform(block):
    bl_shape = block.shape
    for i in range(bl_shape[0]):
        for j in range(bl_shape[1]):
            
def apply_idct(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# Quantization and Inverse Quantization
def quantize(block, quant_matrix):
    return np.round(block / quant_matrix).astype(np.int32)

def dequantize(block, quant_matrix):
    return (block * quant_matrix).astype(np.float32)

# Processing for Compression and Decompression
def process_channel_for_compression(channel, quant_matrix):
    height, width = channel.shape
    padded_height = (height + 7) // 8 * 8
    padded_width = (width + 7) // 8 * 8
    padded_channel = np.pad(channel, ((0, padded_height - height), (0, padded_width - width)), mode='constant', constant_values=0)
    
    compressed_channel = []
    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            block = padded_channel[i:i+8, j:j+8] - 128  # Level shift
            dct_block = apply_dct(block)
            quantized_block = quantize(dct_block, quant_matrix)
            compressed_channel.append(quantized_block)
    return compressed_channel, padded_height, padded_width

def process_channel_for_decompression(compressed_channel, padded_height, padded_width, quant_matrix):
    decompressed_channel = np.zeros((padded_height, padded_width), dtype=np.float32)
    idx = 0
    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            quantized_block = compressed_channel[idx]
            dequantized_block = dequantize(quantized_block, quant_matrix)
            idct_block = apply_idct(dequantized_block) + 128  # Level shift back
            decompressed_channel[i:i+8, j:j+8] = idct_block
            idx += 1
    return np.clip(decompressed_channel[:padded_height, :padded_width], 0, 255).astype(np.uint8)

# JPEG Compression and Decompression Pipeline
def jpeg_compress_decompress(image_path):
    image = plt.imread(image_path)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()
    
    ycbcr_image = rgb_to_ycbcr(image)
    Y, Cb, Cr = subsample(ycbcr_image)

    global quant_matrix
    quant_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                             [12, 12, 14, 19, 26, 58, 60, 55],
                             [14, 13, 16, 24, 40, 57, 69, 56],
                             [14, 17, 22, 29, 51, 87, 80, 62],
                             [18, 22, 37, 56, 68, 109, 103, 77],
                             [24, 35, 55, 64, 81, 104, 113, 92],
                             [49, 64, 78, 87, 103, 121, 120, 101],
                             [72, 92, 95, 98, 112, 100, 103, 99]])

    compressed_Y, h_Y, w_Y = process_channel_for_compression(Y, quant_matrix)
    compressed_Cb, h_Cb, w_Cb = process_channel_for_compression(Cb, quant_matrix)
    compressed_Cr, h_Cr, w_Cr = process_channel_for_compression(Cr, quant_matrix)

    decompressed_Y = process_channel_for_decompression(compressed_Y, h_Y, w_Y, quant_matrix)
    decompressed_Cb = process_channel_for_decompression(compressed_Cb, h_Cb, w_Cb, quant_matrix)
    decompressed_Cr = process_channel_for_decompression(compressed_Cr, h_Cr, w_Cr, quant_matrix)

    decompressed_ycbcr = upsample(decompressed_Y, decompressed_Cb, decompressed_Cr)
    decompressed_image = ycbcr_to_rgb(decompressed_ycbcr)

    plt.imshow(decompressed_image)
    plt.title("Decompressed Image")
    plt.axis('off')
    plt.show()

    return decompressed_image

# Example usage
jpeg_compress_decompress('/home/allemmbinn/Documents/JPEG_Compression/color_image.jpg')
