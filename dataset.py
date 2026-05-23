import os
import pandas as pd

IMAGE_DIR = "urine specimen"

# ✅ create required folders FIRST
os.makedirs("dataset", exist_ok=True)

data = []

def add_range(start, end, color, foam, clarity):
    for i in range(start, end + 1):
        uid = f"U{i:03d}.jpg"
        path = os.path.join(IMAGE_DIR, uid)

        if os.path.exists(path):
            data.append({
                "image": path,
                "color": color,
                "foam": foam,
                "clarity": clarity,
                "label": f"{color}_{foam}_{clarity}"
            })

# (your mappings stay the same...)
# 🟡 pale_yellow
add_range(1, 5, "pale_yellow", "none", "clear")
add_range(6, 6, "pale_yellow", "mild_white", "clear")
add_range(16, 16, "pale_yellow", "moderate_white", "slightly_cloudy")

# 🟡 yellow
add_range(21, 25, "yellow", "none", "clear")
add_range(36, 38, "yellow", "moderate", "cloudy")

# 🟠 dark_yellow
add_range(41, 45, "dark_yellow", "none", "clear")

# 🟤 amber
add_range(61, 65, "amber", "none", "clear")

# 🔴 red_pink
add_range(71, 73, "red_pink", "clear_foam", "clear")
add_range(76, 76, "red_pink", "foam", "cloudy")

# 🟤 brown
add_range(86, 87, "brown", "low", "semi_transparent")

# ✅ SAVE CSV (now safe)
df = pd.DataFrame(data)
df.to_csv("dataset/labels.csv", index=False)

print("Dataset created successfully!")
print("Total images labeled:", len(df))
