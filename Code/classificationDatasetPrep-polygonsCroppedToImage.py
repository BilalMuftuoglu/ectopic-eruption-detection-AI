

import os
import json
import pandas as pd
from PIL import Image
from shutil import copyfile


json_dir = "YOUR_JSON_DIR" #"C:\Users\doguk\Desktop\araproje\ektopik-erupsiyon-bilal-dogukan\ektopik-erupsiyon\dataset\all-json"
image_dir = "YOUR_IMAGE_DIR" #"C:\Users\doguk\Desktop\araproje\ektopik-erupsiyon-bilal-dogukan\ektopik-erupsiyon\dataset\all-image"
excel_file = "YOUR_EXCEL_PATH" #"C:\Users\doguk\Desktop\araproje\ektopik-erupsiyon-bilal-dogukan\ektopik-erupsiyon\dataExcel.xlsx"
output_dir = "YOUR_OUTPUT_DIR" #"C:\Users\doguk\Desktop\araproje\ektopik-erupsiyon-bilal-dogukan\ektopik-erupsiyon\Classification\newTrainAllImages\newTrainAllImages-2clas"

import os
import json
from PIL import Image

def cropImagesprewithperm(json_dir, image_dir, output_dir):
    for image_file in os.listdir(image_dir):
        if image_file.endswith(".png") or image_file.endswith(".jpg") or image_file.endswith(".bmp"):
            print(image_file)

            # Load corresponding JSON file
            json_file = os.path.join(json_dir, os.path.splitext(image_file)[0] + ".json")
        
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Load image
                image_path = os.path.join(image_dir, image_file)
                image = Image.open(image_path)

                # Create folders for each tooth if they don't exist
                for tooth_name in ["55", "65", "75", "85"]:
                    folder_path = os.path.join(output_dir, tooth_name)
                    os.makedirs(folder_path, exist_ok=True)

                # Extract coordinates and crop teeth
                for tooth_data in data["outputs"]["object"]:
                    
                    tooth_name = tooth_data["name"]
                                                
                    if tooth_name in ["55", "65", "75", "85"]:
                        polygon = tooth_data["polygon"]
                        x_values = [polygon[f'x{i}'] for i in range(1, len(polygon)+1) if f'x{i}' in polygon]
                        y_values = [polygon[f'y{i}'] for i in range(1, len(polygon)+1) if f'y{i}' in polygon]
                        left = max(0, min(x_values))   
                        top = max(0, min(y_values))  
                        right = min(image.width, max(x_values)) 
                        bottom = min(image.height, max(y_values)) 
                        if left < right and top < bottom:
                            # Get corresponding permanent tooth
                            permanent_tooth_name = str(int(tooth_name) - 39)
                            permanent_tooth_found = False
                            for perm_tooth_data in data["outputs"]["object"]:
                                if perm_tooth_data["name"] == permanent_tooth_name:
                                    permanent_tooth_found = True
                                    perm_polygon = perm_tooth_data["polygon"]
                                    perm_x_values = [perm_polygon[f'x{i}'] for i in range(1, len(perm_polygon)+1) if f'x{i}' in perm_polygon]
                                    perm_y_values = [perm_polygon[f'y{i}'] for i in range(1, len(perm_polygon)+1) if f'y{i}' in perm_polygon]
                                    perm_left = max(0, min(perm_x_values))   
                                    perm_top = max(0, min(perm_y_values))  
                                    perm_right = min(image.width, max(perm_x_values)) 
                                    perm_bottom = min(image.height, max(perm_y_values)) 
                                    if perm_left < perm_right and perm_top < perm_bottom:
                                        # Expand the cropping coordinates to include both teeth
                                        left = min(left, perm_left)
                                        top = min(top, perm_top)
                                        right = max(right, perm_right)
                                        bottom = max(bottom, perm_bottom)

                                        # Crop and save the image
                                        tooth_image = image.crop((left, top, right, bottom))
                                        tooth_image.save(os.path.join(output_dir, tooth_name, f"{os.path.splitext(image_file)[0]}.png"), "PNG")
                         
def moveImages(excel_file, image_dir, output_dir):
    df = pd.read_excel(excel_file)
    for i in [55,65,75,85]:
        teethFolder=os.path.join(output_dir,str(i))
        for j in os.listdir(teethFolder):
            if os.path.isdir(os.path.join(teethFolder,j)):
                continue
            image_number = j[:3]
            row = int(image_number) - 1
            image_name = df["Ad soyad"][row]
            image_label = df[i][row]
            source_image_path = os.path.join(teethFolder, j)
            destination_image_path = os.path.join(teethFolder,image_label, image_name)
            if(not os.path.exists(os.path.join(teethFolder,image_label))):
                os.makedirs(os.path.join(teethFolder,image_label), exist_ok=True)
                
            copyfile(source_image_path, destination_image_path)
            #delete the original image
            os.remove(source_image_path)

cropImagesprewithperm(json_dir, image_dir, output_dir)
moveImages(excel_file, image_dir, output_dir)