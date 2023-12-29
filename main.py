import pandas as pd
import numpy as np
import matplotlib
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans

st.set_page_config(page_title="Color Palette Extractor")

st.title("COLOR PALETTE EXTRACTOR")
st.text("This tool will extract 3 dominant colors of an image.")
image_uploaded = st.file_uploader("Upload an image: ")
if image_uploaded:
    st.image(image_uploaded)


    btn = st.button("Show color palette")
    X = matplotlib.image.imread(image_uploaded)
    X = X.reshape(-1,3)
    kmeans = KMeans(n_clusters=3, init="k-means++")
    kmeans.fit(X)

    def create_color_palette(dominant_colors, palette_size=(300, 50)):
        # Create an image to display the colors
        palette = Image.new("RGB", palette_size)
        draw = ImageDraw.Draw(palette)

        # Calculate the width of each color swatch
        swatch_width = palette_size[0] // len(dominant_colors)

        # Draw each color as a rectangle on the palette
        for i, color in enumerate(dominant_colors):
            draw.rectangle([i * swatch_width, 0, (i + 1) * swatch_width, palette_size[1]], fill=tuple(color))

        return palette


    def draw_on_image():
        # Create a white canvas
        image = Image.new("RGB", (300, 300), "white")
        draw = ImageDraw.Draw(image)

        # Draw a simple rectangle
        draw.rectangle([(50, 50), (250, 250)], outline="black", width=2)

        return image


    output = create_color_palette(kmeans.cluster_centers_.astype(int))

    # Convert PIL Image to NumPy array
    image_array = np.array(output)

    # Display the image in Streamlit
    st.image(image_array, caption="The 3 dominant colors", use_column_width=True)

