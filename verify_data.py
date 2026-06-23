import struct
import os

def load_and_check(raw_file, meta_file):
    try:
        with open(meta_file, 'r') as f:
            parts = f.read().strip().split()
            w, h = int(parts[0]), int(parts[1])
        
        # Load Raw Data - read first pixel using standard python
        if not os.path.exists(raw_file):
             print(f"Error: {raw_file} does not exist!")
             return None

        file_size = os.path.getsize(raw_file)
        if file_size == 0:
            print(f"Error: {raw_file} is empty!")
            return None
        
        with open(raw_file, 'rb') as f:
            # Read 2 bytes for a 16-bit unsigned short
            bytes_data = f.read(2)
            if len(bytes_data) < 2:
                print(f"Error: Not enough data in {raw_file}")
                return None
            
            # Unpack: '<' = little-endian, 'H' = unsigned short (2 bytes)
            val = struct.unpack('<H', bytes_data)[0]
            
        print(f"\n--- Checking {raw_file} ---")
        print(f"Dimensions: {w} x {h}")
        print(f"First pixel (0,0) value: {val}")
        return val

    except Exception as e:
        print(f"Failed to read {raw_file}: {e}")
        return None

print("=== VERIFYING SERVER DATA (NO NUMPY) ===")
val_red = load_and_check("images/image_B04.raw", "images/image_B04_meta.txt") # Red
val_nir = load_and_check("images/image_B08.raw", "images/image_B08_meta.txt") # NIR

if val_red is not None and val_nir is not None:
    r = float(val_red)
    n = float(val_nir)
    
    print("\n=== CALCULATION CHECK (Pixel 0) ===")
    print(f"Red: {r}")
    print(f"NIR: {n}")
    
    denom = n + r
    if denom == 0:
        ndvi = 0.0
    else:
        ndvi = (n - r) / denom
        
    print(f"Server-Side Python NDVI: {ndvi:.8f}")
