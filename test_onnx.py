import onnxruntime as ort
import numpy as np

# Load session with CUDA or CPU provider
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session = ort.InferenceSession("weights/drone_detector.onnx", providers=providers)

# Create a dummy batch of 2 sliced tiles
dummy_tiles = np.random.randn(2, 3, 640, 640).astype(np.float32)

boxes, scores = session.run(["boxes", "scores"], {"images": dummy_tiles})

print(f"Active Provider: {session.get_providers()[0]}")
print(f"Decoded Boxes Shape:  {boxes.shape}  -> Expected: (2, 34000, 4)")
print(f"Decoded Scores Shape: {scores.shape} -> Expected: (2, 34000, 1)")
print(f"Score Range: min={scores.min():.4f}, max={scores.max():.4f}")