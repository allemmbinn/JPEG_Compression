import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import pickle
from skimage import data
from decoder import jpeg_decode

"""
Standard JPEG Luminance Quantization Matrix
"""
QUANT_MATRIX_Y = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99]
])

"""
Standard JPEG Chrominance Quantization Matrix
"""
QUANT_MATRIX_C = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

"""
Function : Scale a base quantization matrix to a given quality level
Input    : base_matrix (8x8), quality (1–100; higher = better quality / less compression)
Output   : Scaled quantization matrix (8x8, int32), values clamped to [1, 255]
"""
def scale_quant_matrix(base_matrix, quality):
    quality = max(1, min(100, quality))
    scale   = 5000 / quality if quality < 50 else 200 - 2 * quality
    scaled  = np.floor((base_matrix * scale + 50) / 100).astype(np.int32)
    return np.clip(scaled, 1, 255)

# Standard zigzag scan order for an 8x8 block (row, col) pairs
ZIGZAG_ORDER = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
]

"""
Function : Convert RGB Image to YCbCr Image
Input    : RGB Image (H x W x 3, uint8)
Output   : YCbCr Image (H x W x 3, float32)
"""
def rgb_to_ycbcr(image):
    # Transformation Matrix (RGB -> YCbCr)
    T = np.array([
        [ 0.299,  0.587,  0.114],
        [-0.169, -0.331,  0.500],   # +0.500 for B (standard JPEG YCbCr)
        [ 0.500, -0.419, -0.081]
    ])
    img_float   = image.astype(np.float32)
    ycbcr_image = img_float @ T.T + np.array([0, 128, 128], dtype=np.float32)
    return ycbcr_image

"""
Function : Sub-sample the Cb and Cr channels using 4:2:0 scheme
Input    : YCbCr Image (H x W x 3, float32)
Output   : Y, Cb, Cr channels (Cb and Cr at half resolution)
"""
def subsample(ycbcr_image):
    Y  = ycbcr_image[:, :, 0]
    Cb = ycbcr_image[:, :, 1][::2, ::2]
    Cr = ycbcr_image[:, :, 2][::2, ::2]
    return Y, Cb, Cr

"""
Function : Apply 2D DCT to an 8x8 block
Input    : 8x8 numpy array (float)
Output   : DCT coefficient block (8x8)
"""
def apply_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

"""
Function : Quantize DCT block using the standard quantization matrix
Input    : DCT block (8x8), quantization matrix (8x8)
Output   : Quantized integer block (8x8, int32)
"""
def quantize(block, quant_matrix):
    return np.round(block / quant_matrix).astype(np.int32)

"""
Function : Scan an 8x8 quantized block in zigzag order
Input    : Quantized 8x8 block
Output   : List of 64 integers in zigzag order
"""
def zigzag_scan(block):
    return [int(block[r, c]) for r, c in ZIGZAG_ORDER]

"""
Function : Run-Length Encode the AC coefficients of a zigzag-scanned block
Input    : 64-element coefficient list (index 0 is the DC coefficient)
Output   : DC coefficient (int), RLE list of (zero_run, value) tuples
"""
def rle_encode(coefficients):
    dc_coeff  = coefficients[0]
    ac_coeffs = coefficients[1:]
    encoded   = []
    zero_run  = 0
    for val in ac_coeffs:
        if val == 0:
            zero_run += 1
        else:
            encoded.append((zero_run, val))
            zero_run = 0
    encoded.append((0, 0))  # End-of-Block (EOB) marker
    return dc_coeff, encoded

"""
Function : Compress a single image channel using DCT + Quantization + RLE
Input    : Channel array (H x W, float32), quantization matrix (8x8)
Output   : Compressed block list, padded height, padded width
"""
def process_channel_for_compression(channel, quant_matrix):
    height, width = channel.shape
    padded_height = (height + 7) // 8 * 8
    padded_width  = (width  + 7) // 8 * 8
    padded_channel = np.pad(channel,
                            ((0, padded_height - height), (0, padded_width - width)),
                            mode='constant', constant_values=128)
    compressed_channel = []
    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            block       = padded_channel[i:i+8, j:j+8] - 128  # Level shift
            dct_block   = apply_dct(block)
            quant_block = quantize(dct_block, quant_matrix)
            zigzag      = zigzag_scan(quant_block)
            dc, rle     = rle_encode(zigzag)
            compressed_channel.append((dc, rle))
    return compressed_channel, padded_height, padded_width

"""
Function : Estimate the byte size of a compressed channel
Input    : Compressed channel list of (dc, rle) tuples
Output   : Estimated byte count (assuming int16 encoding)
"""
def estimate_channel_size(compressed_channel):
    total = 0
    for dc_coeff, rle in compressed_channel:
        total += 2              # DC coefficient stored as int16 (2 bytes)
        total += len(rle) * 4  # Each (zero_run, value) pair = 2 x int16
    return total

"""
Function : JPEG Encoder - Full compression pipeline
Input    : image_path  - path to input image file
           output_path - path to save compressed data (.pkl)
           quality     - compression quality level (1–100; higher = better quality)
Output   : Compression ratio (float), saves compressed .pkl file
"""
def jpeg_encode(image_path, output_path='compressed.pkl', quality=75):
    image = plt.imread(image_path)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Normalise to H x W x 3 (handle grayscale and RGBA inputs)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    image = image[:, :, :3]

    original_height, original_width = image.shape[:2]
    original_size = original_height * original_width * 3  # bytes for raw RGB

    qmat_Y = scale_quant_matrix(QUANT_MATRIX_Y, quality)
    qmat_C = scale_quant_matrix(QUANT_MATRIX_C, quality)

    print(f"[Encoder] Image       : {image_path}")
    print(f"[Encoder] Quality     : {quality}")
    print(f"[Encoder] Dimensions  : {original_height} x {original_width}")
    print(f"[Encoder] Original    : {original_size / 1024:.2f} KB")

    ycbcr_image             = rgb_to_ycbcr(image)
    Y, Cb, Cr               = subsample(ycbcr_image)
    comp_Y,  h_Y,  w_Y      = process_channel_for_compression(Y,  qmat_Y)
    comp_Cb, h_Cb, w_Cb     = process_channel_for_compression(Cb, qmat_C)
    comp_Cr, h_Cr, w_Cr     = process_channel_for_compression(Cr, qmat_C)

    compressed_size = (estimate_channel_size(comp_Y) +
                       estimate_channel_size(comp_Cb) +
                       estimate_channel_size(comp_Cr))
    ratio = original_size / compressed_size

    print(f"[Encoder] Compressed  : {compressed_size / 1024:.2f} KB")
    print(f"[Encoder] Ratio       : {ratio:.2f}:1")

    # Bundle only what the decoder needs — do NOT store the raw image
    compressed_data = {
        'Y'  : comp_Y,  'h_Y'  : h_Y,  'w_Y'  : w_Y,
        'Cb' : comp_Cb, 'h_Cb' : h_Cb, 'w_Cb' : w_Cb,
        'Cr' : comp_Cr, 'h_Cr' : h_Cr, 'w_Cr' : w_Cr,
        'quant_matrix_Y' : qmat_Y,
        'quant_matrix_C' : qmat_C,
        'original_height': original_height,
        'original_width' : original_width,
    }
    with open(output_path, 'wb') as f:
        pickle.dump(compressed_data, f)
    print(f"[Encoder] Saved to    : {output_path}")

    return ratio


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    image = data.astronaut()  # Load sample color image from skimage (RGB)
    plt.imsave('astronaut.png', image)

    QUALITY_LEVELS = [
        ('Low',    25),
        ('Medium', 50),
        ('High',   75),
    ]

    results = []  # (label, quality, ratio, reconstructed_image)
    for label, quality in QUALITY_LEVELS:
        pkl_path = f'astronaut_q{quality}.pkl'
        ratio    = jpeg_encode('astronaut.png', pkl_path, quality=quality)
        recon    = jpeg_decode(pkl_path)
        results.append((label, quality, ratio, recon))
        print()

    # --- 4-panel comparison: Original | Low | Medium | High ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=13)
    axes[0].axis('off')

    for ax, (label, quality, ratio, recon) in zip(axes[1:], results):
        ax.imshow(recon)
        ax.set_title(f"{label} (Q={quality})\nRatio {ratio:.2f}:1", fontsize=13)
        ax.axis('off')

    plt.suptitle("JPEG Quality Level Comparison", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('astronaut_quality_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[Main] Quality comparison saved to : astronaut_quality_comparison.png")
