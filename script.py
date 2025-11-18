import matplotlib.font_manager as fm
import pprint

# Get all available fonts
all_fonts = fm.fontManager.ttflist

# Create a list of font names
font_names = [font.name for font in all_fonts]
font_names = sorted(list(set(font_names)))  # Remove duplicates and sort

print("Total fonts available:", len(font_names))
print("\n" + "="*50)
print("ALL AVAILABLE FONTS:")
print("="*50)

# Print all fonts
for font in font_names:
    print(f"'{font}'")

print("\n" + "="*50)
print("SEARCHING FOR MONTSERRAT:")
print("="*50)

# Look specifically for Montserrat variants
montserrat_fonts = [font for font in font_names if 'montserrat' in font.lower()]
if montserrat_fonts:
    print("Found Montserrat fonts:")
    for font in montserrat_fonts:
        print(f"  - '{font}'")
else:
    print("No Montserrat fonts found!")

# Also check font files and their properties
print("\n" + "="*50)
print("DETAILED MONTSERRAT INFO:")
print("="*50)

for font in all_fonts:
    if 'montserrat' in font.name.lower():
        print(f"Name: '{font.name}'")
        print(f"File: {font.fname}")
        print(f"Style: {font.style}")
        print(f"Weight: {font.weight}")
        print("-" * 30)