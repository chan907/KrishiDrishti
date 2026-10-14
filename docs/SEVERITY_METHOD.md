# Severity estimation method (PHASE 13)

Classification says *which* disease; severity asks *how much of the visible leaf is affected*. They are different problems.

## Why a classical method
The classification datasets have **no segmentation masks and no severity labels**. A learned severity model would need either a mask dataset (U-Net) or severity-graded labels. We start with the simplest defensible method and make it *measurable*.

## Algorithm (`core/krishi_core/severity.py`)
1. Downscale (max side 512), Gaussian blur, convert to HSV.
2. **Leaf mask**: pixels with hue in the green-yellow-brown range with enough saturation/brightness -> morphological closing -> largest contour -> filled (fills holes left by dark lesions).
3. **Healthy green**: hue 35-95 (OpenCV scale), saturation > 40, value > 40.
4. **Lesion** = leaf mask minus healthy green, cleaned with a morphological opening.
5. `severity % = lesion pixels / leaf pixels x 100`.
6. Refuse (return "not available") when no leaf is found or the leaf covers < 8% of the frame.

## Levels (application-defined, NOT an agronomic standard)
0-10% Mild, 11-30% Moderate, 31-60% High, >60% Severe. Healthy predictions report level "None". Replace with a crop-specific agronomic scale if you adopt one, and cite it.

## Known limitations
- Colour-threshold heuristic: yellowing from nutrient deficiency/senescence counts as "affected".
- Soil or brown backgrounds can be mistaken for leaf/lesion; works best with a single leaf on a plain background.
- Leaf edges in shadow may be mis-segmented.
- Reports visible **area** only - not yield loss.

## Validation (required before trusting numbers)
Annotate 30-100 images (leaf mask + lesion mask) and run `python -m ml.src.severity.validate_severity`; report mean absolute error in percentage points and the failure cases. Unit tests (`core/tests/test_core.py`) check behaviour on synthetic leaves only (monotonic with lesion size, near zero for a clean leaf) - they are *not* a validation on real leaves.

## Upgrade path
B. Train a U-Net on a lesion-mask dataset (e.g. PlantSeg-style data) -> replace `estimate_severity` behind the same function signature.
