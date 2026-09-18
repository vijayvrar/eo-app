import os
import numpy as np
import rasterio
import onnxruntime as ort
from rasterio.env import Env
from rasterio.warp import transform
from rasterio.windows import Window
from pystac_client import Client
from PIL import Image

# Initialize lightweight ONNX session (Uses < 150MB RAM)
SESSION = None

def get_onnx_session():
    global SESSION
    if SESSION is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        SESSION = ort.InferenceSession("edsr.onnx", opts)
    return SESSION

def apply_super_resolution(pil_img):
    """Runs 4x EDSR AI inference via ONNX Runtime without PyTorch RAM overhead."""
    session = get_onnx_session()
    
    # Pre-process image to float32 tensor [1, 3, H, W]
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    img_tensor = np.transpose(img_np, (2, 0, 1))[np.newaxis, ...]

    # Run AI inference
    outputs = session.run(None, {"input": img_tensor})
    output_tensor = outputs[0][0]

    # Post-process array back to uint8 PIL image
    output_np = np.transpose(output_tensor, (1, 2, 0))
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    
    return Image.fromarray(output_np)

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
    crop_size = 128  # 128x128 native patch converts to 512x512 2.5m AI output
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

    # Run ONNX AI Model
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
