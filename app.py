import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

auth_token = "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"

# Create app
app = tk.Tk()
app.geometry("532x622")
app.title("Stable Diffusion AI")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, width=512, height=40, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)


lmain  = ctk.CTkLabel(app,height=512,width=512)
lmain.place(x=10,y=110)

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

def generate():
    with autocast(device):
        # guidance scale is how detailed the image will be
        output = pipe(prompt.get(), guidance_scale=8.5)
        if 'generated' in output:
            # Access the 'generated' key to get the generated image
            image = output['generated'][0]
            image.save('generatedimage.png')    
            img = ImageTk.PhotoImage(image)
            lmain.configure(image=img)
        else:
            # Handle the case when the 'generated' key is not present in the output
            print("Generated image not found in output.")

trigger = ctk.CTkButton(app,height = 40,font=("Arial", 20), text_color="white", fg_color="blue", command=generate )
trigger.configure(text = "Generate")
trigger.place(x=206, y=60)

app.mainloop()
