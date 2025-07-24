from pathlib import Path
from scripts.predict_image import predict_spill_from_image

result = predict_spill_from_image(Path("data/test/oilspill.1591783104.jpg"))
print(result)
