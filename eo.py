import os
import io
from pathlib import Path
import torch
import numpy as np
import rasterio
from rasterio.env import Env
from rasterio.warp import transform
from rasterio.windows import Window
from pystac_client import Client
import matplotlib.pyplot as plt
from PIL import Image
from super_image import EdsrModel, ImageLoader

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MERGE_CONSECUTIVE_DATA_FOR_ALL_DAILY_FILES": "YES",
    "VSI_CACHE": "TRUE",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    sr_model = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=4).to(device)
    sr_model.eval()
    print("✅ Pretrained 4x Super-Resolution AI Model Loaded Successfully!")
except Exception as e:
    print(f"⚠️ Model load fallback: {e}")
    sr_model = None


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
        raise Exception("No suitable low-cloud satellite data found for these coordinates.")

    item = items[0]
    start_time = item.properties.get("datetime")
    assets = item.assets

    red_url = assets["red"].href
    green_url = assets["green"].href
    blue_url = assets["blue"].href

    wide_size = 512
    half_wide = wide_size // 2

    with Env(**GDAL_ENV):
        with rasterio.open(red_url) as src:
            x, y = transform("EPSG:4326", src.crs, [longitude], [latitude])
            row, col = src.index(x[0], y[0])

            r_start = max(0, row - half_wide)
            r_end = min(src.height, row + half_wide)
            c_start = max(0, col - half_wide)
            c_end = min(src.width, col + half_wide)

            window = Window(c_start, r_start, c_end - c_start, r_end - r_start)
            red = src.read(1, window=window)

        with rasterio.open(green_url) as src:
            green = src.read(1, window=window)

        with rasterio.open(blue_url) as src:
            blue = src.read(1, window=window)

    rgb = np.dstack((red, green, blue)).astype(np.float32) / 10000.0
    low, high = np.percentile(rgb, (2, 98))
    if high <= low:
        high = low + 1e-5

    rgb_norm = np.clip((rgb - low) / (high - low), 0, 1)
    rgb_8bit = (rgb_norm * 255).astype(np.uint8)

    pil_img = Image.fromarray(rgb_8bit)

    output_dir = Path("static/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Native 10m Baseline
    native_file_path = output_dir / "sentinel_10m.png"
    pil_img.save(native_file_path)

    # 2. Save 2.5m AI Super-Resolved Image
    sr_file_path = output_dir / "sentinel_sr_2.5m.png"
    if sr_model is not None:
        inputs = ImageLoader.load_image(pil_img).to(device)
        with torch.no_grad():
            preds = sr_model(inputs)
        ImageLoader.save_image(preds, str(sr_file_path))
    else:
        pil_img.save(sr_file_path)

    # 3. Generate Heightmap for Three.js 3D Displacement
    enhanced_img = Image.open(sr_file_path)
    heightmap = enhanced_img.convert("L")
    heightmap_file_path = output_dir / "sentinel_heightmap.png"
    heightmap.save(heightmap_file_path)

    return {
        "image_10m": "/static/outputs/sentinel_10m.png",
        "image_2_5m": "/static/outputs/sentinel_sr_2.5m.png",
        "heightmap": "/static/outputs/sentinel_heightmap.png",
        "observation": start_time,
        "granule": item.id,
        "latitude": latitude,
        "longitude": longitude
    }