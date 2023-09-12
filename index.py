# Imports
import streamlit as st
import numpy as np
from skimage import io
from skimage.transform import radon, iradon, iradon_sart, rescale
import matplotlib.pyplot as plt

# App title
st.title("CT Reconstruction")

st.write("""
A minimal demo of computed tomography image reconstruction, built using 
[streamlit](https://github.com/streamlit/streamlit).

To get started, upload any image as an input or use the example loaded below.
""")

# Image uploader
input_image = st.file_uploader("Upload an alternative input image (optional)", type=['png', 'jpg'])
if input_image is None:
    input_image = "./default_input_image.png"

# Image rescale
cols = st.columns(2)
scale = cols[0].slider("Rescale the image", min_value=0.1, max_value=1.0, value=1.0)
img = io.imread(input_image, as_gray=True)
img = rescale(img, scale=scale, mode='reflect', multichannel=False)

fig, ax = plt.subplots()
fig.colorbar(ax.imshow(img, cmap="gray", vmin=0, vmax=1.0), ax=ax)
cols[1].pyplot(fig, clear_figure=True)
cols[1].caption("Input Image")

# Sinogram
st.write("""
## Sinogram
A sinogram is a graphical representation of raw data in CT. As the image is rotated through various angles, 
the sinogram captures the intensity variations. Each horizontal line in the sinogram corresponds to a single 
rotation angle, and each vertical axis represents the position of the detectors.

For a complete reconstruction, at least 180 degrees must be covered.
""")

cols = st.columns(2)
max_angle = cols[0].slider("Angle Range", min_value=10.0, max_value=180.0, value=150.0)
theta = np.linspace(0.0, max_angle, max(img.shape))
sinogram = radon(img, theta=theta)
fig, ax = plt.subplots()
fig.colorbar(ax.imshow(sinogram, cmap="gray"), ax=ax)
cols[1].pyplot(fig, clear_figure=True)
cols[1].caption("Sinogram")

# Image reconstruction
st.write("""
## Image Reconstruction
We can reconstruct the original image from the sinogram using various algorithms. 
Each has its own advantages and might yield slightly different results.
""")

cols = st.columns(2)
method = cols[0].selectbox("Reconstruction Method", ["Filtered Back Projection", "SART"])
if method == "Filtered Back Projection":
    filter_name = cols[1].selectbox("Filtertype", ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'])
    st.write("""
    Filtered Back Projection (FBP) is a common method for reconstructing images from their projections. 
    Different filters can be applied during this process, affecting the output image's sharpness and noise levels.
    """)
    reconstruction = iradon(sinogram, theta=theta, filter_name=filter_name)
else:  # 'SART'
    st.write("""
    Simultaneous Algebraic Reconstruction Technique (SART) is an iterative method that refines the 
    reconstructed image over several iterations, often yielding better results than FBP for certain scenarios.
    """)
    reconstruction = iradon_sart(sinogram, theta=theta)

# Display results and comparison
st.write("## Results and Comparison")
compare_cols = st.columns(2)

fig, ax = plt.subplots()
fig.colorbar(ax.imshow(img, cmap="gray", vmin=0, vmax=1.0), ax=ax)
compare_cols[0].pyplot(fig, clear_figure=True)
compare_cols[0].caption("Original Image")

fig, ax = plt.subplots()
fig.colorbar(ax.imshow(reconstruction, cmap="gray", vmin=0, vmax=1.0), ax=ax)
compare_cols[1].pyplot(fig, clear_figure=True)
compare_cols[1].caption("Reconstructed Image")
