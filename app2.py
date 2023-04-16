import tkinter as tk
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageTk
from tkinter import ttk

# Create the main window
root = tk.Tk()
root.title("Image Generator")
root.geometry("400x400")

# Load the model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Function to generate the image
def generate_image():
    # Disable generate button and prompt entry during image generation
    generate_button.config(state=tk.DISABLED)
    prompt_entry.config(state=tk.DISABLED)
    
    # Start progress bar
    progress_bar.start()
    
    prompt_text = prompt_entry.get("1.0", tk.END).strip()  # Get the text from the prompt entry
    image = pipe(prompt_text).images[0]
    image.save("generated_image.png")  # Save the generated image
    image = Image.open("generated_image.png")  # Open the generated image
    image = image.resize((300, 300))  # Resize the image
    image = ImageTk.PhotoImage(image)  # Convert the image to a PhotoImage object
    image_label.config(image=image)  # Update the image label
    image_label.image = image  # Store the image object in a label attribute
    
    # Stop progress bar and enable generate button and prompt entry after image generation
    progress_bar.stop()
    generate_button.config(state=tk.NORMAL)
    prompt_entry.config(state=tk.NORMAL)

# Function to generate image when Enter key is pressed in prompt entry
def generate_image_on_enter(event):
    generate_image()

# Create input prompt entry
prompt_label = tk.Label(root, text="Enter Prompt:")
prompt_label.pack(pady=10)
prompt_entry = tk.Text(root, height=1, width=40)
prompt_entry.pack(pady=5)
prompt_entry.bind("<Return>", generate_image_on_enter)  # Bind Enter key to generate_image_on_enter function

# Create generate button
generate_button = tk.Button(root, text="Generate Image", command=generate_image)
generate_button.pack(pady=5)

# Create progress bar
progress_bar = ttk.Progressbar(root, mode="indeterminate")
progress_bar.pack()

# Create image label
image_label = tk.Label(root)
image_label.pack()

# Start the UI event loop
root.mainloop()
