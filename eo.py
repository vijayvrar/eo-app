# eo.py
import os
import numpy as np
import rasterio
import onnxruntime as ort
from rasterio.env import Env
from rasterio.warp import transform
from rasterio.windows import Window
from pystac_client import Client
from PIL import Image

SESSION = None

def get_onnx_session():
    """Lazy load ONNX session to minimize startup RAM."""
    global SESSION
    if SESSION is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        SESSION = ort.InferenceSession("edsr.onnx", opts)
    return SESSION

def apply_super_resolution_tiled(pil_img, tile_size=64):
    """Processes image in 64x64 tiles to strictly respect Render's 512MB RAM cap."""
    session = get_onnx_session()
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    h, w, c = img_np.shape

    scale = 4
    out_h, out_w = h * scale, w * scale
    output_img = np.zeros((out_h, out_w, c), dtype=np.float32)

    # Process via small patches
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img_np[y:y+tile_size, x:x+tile_size, :]
            th, tw, _ = tile.shape

            # Handle edge boundary padding
            if th < tile_size or tw < tile_size:
                padded_tile = np.zeros((tile_size, tile_size, c), dtype=np.float32)
                padded_tile[:th, :tw, :] = tile
                tile = padded_tile

            tile_tensor = np.transpose(tile, (2, 0, 1))[np.newaxis, ...]
            
            # Execute ONNX model
            sr_tile = session.run(None, {"input": tile_tensor})[0][0]
            sr_tile = np.transpose(sr_tile, (1, 2, 0))

            # Crop padding and assign to output array
            sr_tile_valid = sr_tile[:th*scale, :tw*scale, :]
            output_img[y*scale:(y+th)*scale, x*scale:(x+tw)*scale, :] = sr_tile_valid

    output_np = np.clip(output_img * 255.0, 0, 255).astype(np.uint8)
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
    crop_size = 128  # Yields 512x512 enhanced output
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

    # Run Tiled ONNX Model
    sr_img = apply_super_resolution_tiled(pil_img)
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
