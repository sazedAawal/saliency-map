import cv2
import numpy as np

# Load image
image = cv2.imread('./image.jpg')

# Create saliency detector (spectral residual — fast, good for general use)
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
success, saliency_map = saliency.computeSaliency(image)

# saliency_map is float32 in [0, 1] — scale to 0-255 for visualization
saliency_map = (saliency_map * 255).astype(np.uint8)

# Apply a heatmap color scheme
heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

# Blend heatmap with original image
overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

cv2.imwrite('saliency_heatmap.jpg', heatmap)
cv2.imwrite('saliency_overlay.jpg', overlay)