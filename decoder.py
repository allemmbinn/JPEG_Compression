import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fftpack import idct
import pickle


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
Function : Apply 2D Inverse DCT to an 8x8 block
Input    : DCT coefficient block (8x8)
Output   : Reconstructed spatial block (8x8, float32)
"""
def apply_idct(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

"""
Function : Dequantize a quantized DCT block
Input    : Quantized block (8x8, int32), quantization matrix (8x8)
Output   : Dequantized coefficient block (8x8, float32)
"""
def dequantize(block, quant_matrix):
    return (block * quant_matrix).astype(np.float32)

"""
Function : Run-Length Decode the AC coefficients of a block
Input    : DC coefficient (int), RLE list of (zero_run, value) tuples
Output   : Full list of 64 coefficients (DC + 63 AC) in zigzag order
"""
def rle_decode(dc_coeff, rle_encoded):
    ac_coeffs = []
    for zero_run, value in rle_encoded:
        if zero_run == 0 and value == 0:  # End-of-Block marker
            break
        ac_coeffs.extend([0] * zero_run)
        ac_coeffs.append(value)
    ac_coeffs.extend([0] * (63 - len(ac_coeffs)))  # Pad remaining ACs to zero
    return [dc_coeff] + ac_coeffs

"""
Function : Reconstruct an 8x8 block from zigzag-ordered coefficients
Input    : List of 64 integers in zigzag order
Output   : 8x8 numpy array (int32)
"""
def inverse_zigzag(coefficients):
    block = np.zeros((8, 8), dtype=np.int32)
    for idx, (r, c) in enumerate(ZIGZAG_ORDER):
        block[r, c] = coefficients[idx]
    return block

"""
Function : Upsample Cb and Cr channels back to full resolution (4:2:0 reverse)
Input    : Y (H x W), Cb (H/2 x W/2), Cr (H/2 x W/2)
Output   : Stacked YCbCr image (H x W x 3)
"""
def upsample(Y, Cb, Cr):
    Cb_up = np.repeat(np.repeat(Cb, 2, axis=0), 2, axis=1)
    Cr_up = np.repeat(np.repeat(Cr, 2, axis=0), 2, axis=1)
    return np.dstack((Y, Cb_up, Cr_up))

"""
Function : Convert YCbCr Image back to RGB Image
Input    : YCbCr Image (H x W x 3, float32)
Output   : RGB Image (H x W x 3, uint8)
"""
def ycbcr_to_rgb(image):
    # Inverse Transformation Matrix (YCbCr -> RGB)
    T = np.array([
        [1.0,  0.000,  1.402],
        [1.0, -0.344, -0.714],
        [1.0,  1.772,  0.000]
    ])
    shifted   = image.astype(np.float32) - np.array([0, 128, 128], dtype=np.float32)
    rgb_image = shifted @ T.T
    return np.clip(rgb_image, 0, 255).astype(np.uint8)

"""
Function : Decompress a single image channel
Input    : Compressed block list, padded height, padded width, quantization matrix
Output   : Decompressed channel (padded_height x padded_width, uint8)
"""
def process_channel_for_decompression(compressed_channel, padded_height, padded_width, quant_matrix):
    decompressed_channel = np.zeros((padded_height, padded_width), dtype=np.float32)
    idx = 0
    for i in range(0, padded_height, 8):
        for j in range(0, padded_width, 8):
            dc_coeff, rle         = compressed_channel[idx]
            coefficients          = rle_decode(dc_coeff, rle)
            quant_block           = inverse_zigzag(coefficients)
            dequant_block         = dequantize(quant_block, quant_matrix)
            idct_block            = apply_idct(dequant_block) + 128  # Undo level shift
            decompressed_channel[i:i+8, j:j+8] = idct_block
            idx += 1
    return np.clip(decompressed_channel, 0, 255).astype(np.uint8)

"""
Function : JPEG Decoder - Full decompression pipeline
Input    : input_path        - path to compressed .pkl file
           output_path       - path to save decoded image (PNG)
           original_path     - (optional) path to original image for comparison plot
           compare_path      - path to save original vs decoded comparison (PNG)
Output   : Reconstructed RGB image (H x W x 3, uint8)
"""
def jpeg_decode(input_path='compressed.pkl',
                output_path='decoded.png',
                original_path=None,
                compare_path='comparison.png'):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    comp_Y  = data['Y'];   h_Y  = data['h_Y'];   w_Y  = data['w_Y']
    comp_Cb = data['Cb'];  h_Cb = data['h_Cb'];  w_Cb = data['w_Cb']
    comp_Cr = data['Cr'];  h_Cr = data['h_Cr'];  w_Cr = data['w_Cr']
    quant_matrix_Y  = data['quant_matrix_Y']
    quant_matrix_C  = data['quant_matrix_C']
    original_height = data['original_height']
    original_width  = data['original_width']

    print(f"[Decoder] Loaded from : {input_path}")
    print(f"[Decoder] Dimensions  : {original_height} x {original_width}")

    Y  = process_channel_for_decompression(comp_Y,  h_Y,  w_Y,  quant_matrix_Y)
    Cb = process_channel_for_decompression(comp_Cb, h_Cb, w_Cb, quant_matrix_C)
    Cr = process_channel_for_decompression(comp_Cr, h_Cr, w_Cr, quant_matrix_C)

    # Trim decompressed channels to correct pre-upsample dimensions
    cb_h = (original_height + 1) // 2
    cb_w = (original_width  + 1) // 2
    Y    = Y[:original_height, :original_width]
    Cb   = Cb[:cb_h, :cb_w]
    Cr   = Cr[:cb_h, :cb_w]

    ycbcr_image   = upsample(Y, Cb, Cr)
    ycbcr_image   = ycbcr_image[:original_height, :original_width, :]  # trim after upsample
    reconstructed = ycbcr_to_rgb(ycbcr_image.astype(np.float32))

    print(f"[Decoder] Reconstruction complete.")

    # Save decoded image
    plt.imsave(output_path, reconstructed)
    print(f"[Decoder] Decoded image saved to : {output_path}")

    # Save side-by-side comparison (requires original image path)
    if original_path is not None:
        original_image = plt.imread(original_path)
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        if len(original_image.shape) == 2:
            original_image = np.stack([original_image, original_image, original_image], axis=2)
        original_image = original_image[:, :, :3]

        is_gray = np.all(original_image[:, :, 0] == original_image[:, :, 1])
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_image, cmap='gray' if is_gray else None)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        axes[1].imshow(reconstructed, cmap='gray' if is_gray else None)
        axes[1].set_title("JPEG Decoded Image")
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Decoder] Comparison plot saved  : {compare_path}")

    return reconstructed


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    jpeg_decode('cameraman_compressed.pkl',
                output_path='cameraman_decoded.png',
                original_path='cameraman.png',
                compare_path='cameraman_comparison.png')
