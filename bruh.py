import os
import glob
yaml_files = glob.glob('configs/*/*/*/*')
print(len(yaml_files))
for yaml_file in yaml_files:
    with open(yaml_file) as f:
        content = f.read().replace("/content/drive/MyDrive/SuperPixels/","./")
    with open(yaml_file,"w") as f:
            f.write(content)

