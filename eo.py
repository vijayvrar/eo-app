import os
import uuid
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
    """Lazy-loads ONNX session with single-thread restriction to prevent high CPU load."""
    global SESSION
    if SESSION is None:
        onnx_path = os.path.join(os.path.dirname(__file__), "edsr.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"CRITICAL: edsr.onnx weight file missing at {onnx_path}")
            
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        SESSION = ort.InferenceSession(onnx_path, opts)
    return SESSION

def apply_super_resolution_tiled(pil_img, tile_size=64):
    """Processes image in fast 64x64 patches via ONNX Runtime."""
    session = get_onnx_session()
    
    img_np = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    h, w, c = img_np.shape

    scale = 4
    out_h, out_w = h * scale, w * scale
    output_img = np.zeros((out_h, out_w, c), dtype=np.float32)

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = img_np[y:y+tile_size, x:x+tile_size, :]
            th, tw, _ = tile.shape

            if th < tile_size or tw < tile_size:
                padded_tile = np.zeros((tile_size, tile_size, c), dtype=np.float32)
                padded_tile[:th, :tw, :] = tile
                tile = padded_tile

            tile_tensor = np.transpose(tile, (2, 0, 1))[np.newaxis, ...]
            
            input_name = session.get_inputs()[0].name
            sr_tile = session.run(None, {input_name: tile_tensor})[0][0]
            sr_tile = np.transpose(sr_tile, (1, 2, 0))

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
        raise Exception("No cloud-free satellite image found for these coordinates.")

    item = items[0]
    assets = item.assets
    
    # 256x256 crop window yields 2.56km x 2.56km scene coverage (Ideal balance for free tier CPU speed)
    crop_size = 256
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
    if high <= low: 
        high = low + 1e-5
    rgb_norm = np.clip((rgb - low) / (high - low), 0, 1)
    rgb_8bit = (rgb_norm * 255).astype(np.uint8)

    pil_img = Image.fromarray(rgb_8bit)

    # Prepare save outputs
    out_dir = os.path.join("static", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    req_id = uuid.uuid4().hex[:8]
    native_filename = f"sentinel_10m_{req_id}.png"
    sr_filename = f"sentinel_2.5m_{req_id}.png"

    # Save Native 10m
    native_path = os.path.join(out_dir, native_filename)
    pil_img.save(native_path)

    # Execute Fast Tiled ONNX Model -> Sharp 1024x1024 Enhanced Scene
    sr_img = apply_super_resolution_tiled(pil_img)
    sr_path = os.path.join(out_dir, sr_filename)
    sr_img.save(sr_path)

    return {
        "image_10m": f"/static/outputs/{native_filename}",
        "image_2_5m": f"/static/outputs/{sr_filename}",
        "observation": item.properties.get("datetime"),
        "granule": item.id,
        "latitude": latitude,
        "longitude": longitude
    }
