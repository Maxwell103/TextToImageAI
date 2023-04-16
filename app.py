import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create app
app = tk.TK()
app.geometry("532x622")
app.title("Stable Diffusion AI")
ctk.set_appearance_mode("dark")

app.mainloop()