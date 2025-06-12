import os
import zipfile
import shutil
import threading
from PIL import Image, ImageTk, ImageEnhance, ImageOps
import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk
import webbrowser


def gif2grid_zip(gif_path, frame_start, frame_end, output_dir, apply_filter, update_status, update_progress):
    try:
        gif = Image.open(gif_path)

        if gif.width != gif.height:
            raise ValueError("GIF must be square (1:1 ratio)")

        total_frames = gif.n_frames
        if frame_end > total_frames:
            frame_end = total_frames

        tile_size = gif.width // 3

        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

       
        for row in range(3):
            for col in range(3):
                os.makedirs(os.path.join(output_dir, f"tile-{row}-{col}"), exist_ok=True)

        for frame_number in range(frame_start, frame_end):
            gif.seek(frame_number)
            frame = gif.convert('RGBA')

            for row in range(3):
                for col in range(3):
                    left = col * tile_size
                    upper = row * tile_size
                    right = left + tile_size
                    lower = upper + tile_size

                    tile = frame.crop((left, upper, right, lower))

                    if apply_filter == "Grayscale":
                        tile = ImageOps.grayscale(tile).convert("RGBA")
                    elif apply_filter == "Tint":
                        tile = ImageEnhance.Color(tile).enhance(1.5)

                    folder = os.path.join(output_dir, f"tile-{row}-{col}")
                    tile.save(os.path.join(folder, f"frame-{frame_number}.png"))

            update_progress((frame_number - frame_start + 1) / (frame_end - frame_start) * 100)

        zip_path = os.path.join(output_dir, "gif_tiles_export.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(".png"):
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, output_dir)
                        zipf.write(abs_path, rel_path)

        update_status(f"✅ Done! Output zipped to: {zip_path}")
        webbrowser.open(output_dir)

    except Exception as e:
        update_status(f"❌ Error: {str(e)}")


def run_gui():
    def select_gif():
        filepath = filedialog.askopenfilename(filetypes=[("GIF files", "*.gif")])
        if filepath:
            gif_path_var.set(filepath)
            status_label.configure(text=f"📁 Selected: {os.path.basename(filepath)}")
            show_preview(filepath)

    def show_preview(gif_path):
        gif = Image.open(gif_path)
        gif.seek(0)
        frame = gif.convert("RGBA").resize((120, 120))
        frame_img = ImageTk.PhotoImage(frame)
        preview_label.configure(image=frame_img)
        preview_label.image = frame_img

    def select_output_dir():
        path = filedialog.askdirectory()
        if path:
            output_dir_var.set(path)

    def start_processing():
        gif_path = gif_path_var.get()
        output_dir = output_dir_var.get()
        if not gif_path or not output_dir:
            status_label.configure(text="❌ Select both a GIF and an output folder.")
            return

        try:
            start_frame = int(start_frame_entry.get())
            end_frame = int(end_frame_entry.get())
            if start_frame < 0 or end_frame <= start_frame:
                raise ValueError
        except ValueError:
            status_label.configure(text="❌ Invalid frame range.")
            return

        filter_type = filter_menu.get()

        status_label.configure(text="⏳ Processing...")
        progress_bar.set(0)

        thread = threading.Thread(target=gif2grid_zip, args=(
            gif_path, start_frame, end_frame, output_dir, filter_type, status_label.configure, progress_bar.set))
        thread.start()

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("GIF to Grid Splitter PRO")
    app.geometry("560x580")
    app.resizable(False, False)

    global gif_path_var, output_dir_var, start_frame_entry, end_frame_entry, filter_menu, progress_bar, status_label, preview_label

    gif_path_var = ctk.StringVar()
    output_dir_var = ctk.StringVar()

    ctk.CTkLabel(app, text="GIF to Grid Splitter PRO", font=("Arial", 22, "bold")).pack(pady=15)

    ctk.CTkButton(app, text="📂 Select GIF", command=select_gif).pack(pady=5)
    ctk.CTkButton(app, text="📁 Select Output Folder", command=select_output_dir).pack(pady=5)

    preview_label = ctk.CTkLabel(app, text="")
    preview_label.pack(pady=5)

    ctk.CTkLabel(app, text="Start Frame:").pack()
    start_frame_entry = ctk.CTkEntry(app, width=120, justify="center")
    start_frame_entry.insert(0, "0")
    start_frame_entry.pack()

    ctk.CTkLabel(app, text="End Frame:").pack()
    end_frame_entry = ctk.CTkEntry(app, width=120, justify="center")
    end_frame_entry.insert(0, "100")
    end_frame_entry.pack()

    ctk.CTkLabel(app, text="Filter:").pack(pady=5)
    filter_menu = ctk.CTkOptionMenu(app, values=["None", "Grayscale", "Tint"])
    filter_menu.pack()

    ctk.CTkButton(app, text=" Start", command=start_processing).pack(pady=15)

    progress_bar = ctk.CTkProgressBar(app, width=400)
    progress_bar.set(0)
    progress_bar.pack(pady=5)

    status_label = ctk.CTkLabel(app, text="Select a GIF and output folder.", wraplength=480, justify="center")
    status_label.pack(pady=5)

    ctk.CTkLabel(app, text="GUI and cleanup by Notriced — Main code by golden.axolotl", font=("Arial", 12), text_color="gray").pack(side="bottom", pady=5)

    app.mainloop()

run_gui()
