from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk
import customtkinter as ctk
from testForGUI import testOnSingleImage, classify_image

original_image = None
resized_image = None
file_path = None

def open_image():
    global file_path
    file_path = filedialog.askopenfilename(title="Open Image File", filetypes=[("Image files", ".bmp .jpg .jpeg .png")])
    if file_path:
        display_image(file_path)

def display_image(file_path):
    global original_image, resized_image

    original_image = Image.open(file_path)
    resize_image_to_fit_window()

    status_label.config(text=f"Image loaded: {file_path}")
    toothDetectionButton.grid(row=2, column=0, padx=20, pady=10)

def resize_image_to_fit_window(event=None):
    global original_image, resized_image

    if original_image:
        # Get the current window size minus some padding for the other widgets
        window_width = canvas.winfo_width()
        window_height = canvas.winfo_height()

        # Determine the new image size
        image_ratio = original_image.width / original_image.height
        window_ratio = window_width / window_height

        if window_ratio > image_ratio:
            new_height = window_height  # Use full canvas height
            new_width = int(new_height * image_ratio)
        else:
            new_width = window_width  # Use full canvas width
            new_height = int(new_width / image_ratio)

        resized_image = original_image.resize((new_width, new_height))
        photo = ImageTk.PhotoImage(resized_image)

        # Clear the canvas and display the new image
        canvas.delete("all")
        canvas.create_image((window_width - new_width) // 2, (window_height - new_height) // 2, anchor='nw', image=photo)
        canvas.image = photo

    adjust_minimum_size()

def adjust_minimum_size():
    # Determine minimum width and height required for the UI elements
    min_width = max(open_button.winfo_reqwidth(), canvas.winfo_reqwidth()) + 40
    min_height = (
        open_button.winfo_reqheight()
        + canvas.winfo_reqheight()
        + status_label.winfo_reqheight()
        + result_label.winfo_reqheight()
        + toothDetectionButton.winfo_reqheight()
        + 80  # Adding some padding
    )

    root.minsize(min_width, min_height)

def segmentation():
    new_file, croppedImagesPaths = testOnSingleImage(file_path)
    display_image(new_file)
    classification(croppedImagesPaths)
    toothDetectionButton.grid_forget()

def classification(croppedImagesPaths):
    teethCropped = []
    for path in croppedImagesPaths:
        teethCropped.append((path.split("_")[-2], path.split("_")[-1].split(".")[0]))
    text = "Tespit edilen diş ikilileri:"    
    for el in teethCropped:
        text = text + el[0] + "-" + el[1] + ", "
    status_label.config(text=text)
    text2 = ""
    for path in croppedImagesPaths:
        tooth = path.split("_")[-1].split(".")[0]
        result = classify_image(path, tooth, True)
        
        if result[0]["label"] == "hasta":
            result = classify_image(path, tooth, False)
        
        text2 = text2 + path.split("_")[-1].split(".")[0] + ": " + str(result[0]["label"]).capitalize() + " %" + str(f'{(result[0]["score"] * 100):.2f}') + "\n"
    result_label.config(text=text2)

root = tk.Tk()
root.title("Ektopik Erupsiyon Tespit Programı")
root.geometry("800x600")  # Set a default window size
#root.attributes("-fullscreen", True)

open_button = tk.Button(root, text="Görüntü Seç", command=open_image)
open_button.grid(row=0, column=0, padx=20, pady=10)

canvas = tk.Canvas(root, bg=root.cget('bg'))  # Set canvas background to match root background
canvas.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")

status_label = tk.Label(root, text="", padx=20, pady=10)
status_label.grid(row=3, column=0)
result_label = tk.Label(root, text="", padx=20, pady=10)
result_label.grid(row=4, column=0)

toothDetectionButton = tk.Button(root, text="Hastalık Tespit Et", command=segmentation)
toothDetectionButton.grid(row=2, column=0, padx=20, pady=0)
toothDetectionButton.grid_remove()

root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.bind("<Configure>", resize_image_to_fit_window)

root.mainloop()
