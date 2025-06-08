import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    data = np.load(file_path)
    st.write(f"Loaded {file_path.name} with shape {data.shape}, dtype {data.dtype}")
    return data

def calculate_ndvi(image, red_band=3, nir_band=7):
    RED = image[red_band].astype(float)
    NIR = image[nir_band].astype(float)
    ndvi = (NIR - RED) / (NIR + RED + 1e-6)
    return ndvi

def segment_rice_field(ndvi, threshold=None):
    if threshold is None:
        threshold = ndvi.min() + 0.4 * (ndvi.max() - ndvi.min())
        st.write(f"Auto NDVI threshold set to: {threshold:.4f}")
    mask = ndvi > threshold
    return mask, threshold

def get_rgb(image, bands=[2,1,0]):
    rgb = image[bands].astype(float)
    mx = rgb.max(axis=(1,2), keepdims=True)
    mn = rgb.min(axis=(1,2), keepdims=True)
    rgb_norm = (rgb - mn) / (mx - mn + 1e-6)
    rgb_img = np.transpose(rgb_norm, (1,2,0))
    return rgb_img

def plot_results(rgb, ndvi, rice_mask):
    fig, axs = plt.subplots(1,3, figsize=(18,6))

    axs[0].imshow(rgb)
    axs[0].set_title("RGB Image")
    axs[0].axis('off')

    im1 = axs[1].imshow(ndvi, cmap='RdYlGn')
    axs[1].set_title("NDVI")
    axs[1].axis('off')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    axs[2].imshow(rice_mask, cmap='gray')
    axs[2].set_title("Rice Field Mask")
    axs[2].axis('off')

    st.pyplot(fig)

def process_uploaded_file(uploaded_file):
    data = np.load(uploaded_file)
    image = data[0]  
    ndvi = calculate_ndvi(image)
    rice_mask, threshold = segment_rice_field(ndvi)
    rgb_img = get_rgb(image)
    plot_results(rgb_img, ndvi, rice_mask)

def main():
    st.set_page_config(page_title="NDVI Segmentation", layout="wide")
    st.title("Rice Field Segmentation from Satellite Data")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.image("about1.png", use_container_width=True)
    st.sidebar.info("© 2025 - Digital Image Processing | Group 3 2023F")

    uploaded_file = st.file_uploader("Upload file .npy Sentinel-2 data", type=["npy"])
    if uploaded_file is not None:
        process_uploaded_file(uploaded_file)
    else:
        st.info("Please upload a Sentinel-2 .npy file to start processing.")

if __name__ == "__main__":
    main()
