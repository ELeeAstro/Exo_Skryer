"""
Generate the fleck logo!
"""

import os
from PIL import Image

# Save the logo here:
logo_dir = os.path.dirname(__file__)
png_path = os.path.join(logo_dir, 'logo.png')
ico_path = os.path.join(logo_dir, 'logo.ico')

# Convert the PNG into an ICO file:
img = Image.open(png_path)
img.save(ico_path)
