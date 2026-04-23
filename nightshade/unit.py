import glob, os

print(glob.glob(os.path.join("selected_data", "*.p")))
print(glob.glob(os.path.join("poisoned_outputs", "*.p")))
