import os
import numpy as np
import rasterio
from rasterio.env import Env
from rasterio.warp import transform
from rasterio.windows import Window
from pystac_client import Client
from PIL import Image

def apply_super_resolution(pil_img):
    """Uses high-quality Lanczos resampling as a lightweight web fallback."""
    width, height = pil_img.size
    # 4x upscale without heavy PyTorch memory overhead
    return pil_img.resize((width * 4, height * 4), Image.Resampling.LANCZOS)

def get_satellite_image(latitude, longitude):
    catalog = Client.open("https://earth-search.aws.element84.com/v1")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [longitude, latitude]},
        datetime="2025-01-01/2026-12-31",
        query={"eo:cloud_cover": {"lt": 20}},
        max_items=1
    )

    items = list(search.items())
    if not items:
        raise Exception("No suitable satellite image found.")

    item = items[0]
    assets = item.assets
    crop_size = 128  # Compact window size for fast server response
    half_crop = crop_size // 2

    with Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with rasterio.open(assets["red"].href) as src:
            x, y = transform("EPSG:4326", src.crs, [longitude], [latitude])
            row, col = src.index(x[0], y[0])
            
            r_start, r_end = max(0, row - half_crop), min(src.height, row + half_crop)
            c_start, c_end = max(0, col - half_crop), min(src.width, col + half_crop)
            window = Window(c_start, r_start, c_end - c_start, r_end - r_start)
            
            red = src.read(1, window=window)

        with rasterio.open(assets["green"].href) as src:
            green = src.read(1, window=window)

        with rasterio.open(assets["blue"].href) as src:
            blue = src.read(1, window=window)

    rgb = np.dstack((red, green, blue)).astype(np.float32) / 10000.0
    low, high = np.percentile(rgb, (2, 98))
    if high <= low: high = low + 1e-5
    rgb_norm = np.clip((rgb - low) / (high - low), 0, 1)
    rgb_8bit = (rgb_norm * 255).astype(np.uint8)

    pil_img = Image.fromarray(rgb_8bit)

    os.makedirs("static/outputs", exist_ok=True)
    
    native_path = "static/outputs/sentinel_10m.png"
    pil_img.save(native_path)

    sr_img = apply_super_resolution(pil_img)
    sr_path = "static/outputs/sentinel_sr_2.5m.png"
    sr_img.save(sr_path)

    return {
        "image_10m": "/static/outputs/sentinel_10m.png",
        "image_2_5m": "/static/outputs/sentinel_sr_2.5m.png",
        "observation": item.properties.get("datetime"),
        "granule": item.id,
        "latitude": latitude,
        "longitude": longitude
    }
