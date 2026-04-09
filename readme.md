# JPEG Compression Pipeline in Python

## Overview

Final project for **Multimedia Systems and Applications**.

A from-scratch implementation of the JPEG compression pipeline in Python. The encoder and decoder are fully separated, and no built-in JPEG libraries are used — every stage of the standard is implemented manually.

**Author:** 

1. Allen Emmanuel Binny [21EC39035]
2. Ivin Roy             [21EC37021]
3. Aritra Kundu         [21EC37030]

---

## Project Structure

```
JPEG_Compression/
├── main.py       # Encoder — full compression pipeline
├── decoder.py    # Decoder — full decompression pipeline
└── readme.md
```

---

## Pipeline

### Encoder (`main.py`)

| Step | Description |
|------|-------------|
| 1. RGB → YCbCr | Separates luminance (Y) from chrominance (Cb, Cr) using the standard JPEG colour matrix |
| 2. 4:2:0 Chroma Sub-sampling | Downsamples Cb and Cr to half resolution — exploits lower human sensitivity to colour detail |
| 3. 8×8 Block DCT | Applies a 2D Discrete Cosine Transform to each 8×8 block, concentrating energy into low-frequency coefficients |
| 4. Quantization | Divides DCT coefficients by a quantization matrix and rounds to integers, discarding imperceptible high-frequency detail |
| 5. Zigzag Scan | Reorders the 8×8 block into a 1D sequence that places low-frequency coefficients first |
| 6. RLE | Run-Length Encodes the AC coefficients, compressing runs of zeros efficiently |

Compressed data is serialised to a `.pkl` file containing everything the decoder needs (channel data, quantization matrices, original dimensions).

### Decoder (`decoder.py`)

Reverses each step: RLE decode → inverse zigzag → dequantize → IDCT → chroma upsample → YCbCr → RGB.

---

## Quantization Matrices

Two separate standard JPEG quantization matrices are used:

- **Luminance (Y)** — finer quantization, preserving brightness detail perceived by the human eye.
- **Chrominance (Cb, Cr)** — coarser quantization, exploiting lower human sensitivity to colour resolution.

Both matrices are scaled by a **quality factor** before use (see Quality Levels below).

---

## Quality Levels

Quality is controlled by a single integer `quality` (1–100, higher = better).  
The base matrices are scaled using the standard JPEG formula:

```
scale = 5000 / quality        if quality < 50
scale = 200 - 2 × quality     if quality >= 50

quant_matrix = clip(floor((base × scale + 50) / 100), 1, 255)
```

| Quality | Compressed Size | Ratio  | Character |
|---------|----------------|--------|-----------|
| 25 (Low) | ~114 KB | ~6.76:1 | Visible block artefacts, maximum compression |
| 50 (Medium) | ~158 KB | ~4.85:1 | Balanced — good quality with strong compression |
| 75 (High) | ~218 KB | ~3.52:1 | Near-transparent loss, moderate compression |

*(Measured on the 512×512 colour astronaut image, 768 KB raw.)*

---

## Usage

### Encode at a specific quality level

```python
from main import jpeg_encode

jpeg_encode('image.png', output_path='compressed.pkl', quality=75)
```

### Decode

```python
from decoder import jpeg_decode

reconstructed = jpeg_decode('compressed.pkl', output_path='decoded.png')
```

### Run the full quality comparison

```bash
python main.py
```

This encodes the sample image at quality levels 25, 50, and 75, then saves a 4-panel comparison to `cameraman_quality_comparison.png`.

---

## Dependencies

```
numpy
scipy
matplotlib
scikit-image
```

Install with:

```bash
pip install numpy scipy matplotlib scikit-image
```

---

## References

- JPEG Standard (ITU-T T.81 / ISO 10918-1)
- [YCbCr colour space](https://en.wikipedia.org/wiki/YCbCr)
- [Discrete Cosine Transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform)
- [JPEG quantization matrices](https://www.impulseadventure.com/photo/jpeg-quantization.html)
