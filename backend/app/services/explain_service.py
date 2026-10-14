from __future__ import annotations

import numpy as np
from krishi_core.gradcam import make_gradcam, overlay_heatmap


class ExplainService:
    def gradcam_overlay(self, model, batch, rgb_img, class_index: int):
        cam, _ = make_gradcam(model, batch, class_index)
        return overlay_heatmap(np.asarray(rgb_img), cam)
