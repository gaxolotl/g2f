import os
import zipfile
import shutil
import threading
from PIL import Image, ImageTk, ImageEnhance, ImageOps
import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk
import webbrowser


def gif2grid_zip(gif_path, frame_start, frame_end, output_dir, apply_filter, update_status, update_progress, zip_only=False):
    try:
        gif = Image.open(gif_path)

        if gif.width != gif.height:
            raise ValueError("GIF must be square (1:1 ratio)")

        total_frames = gif.n_frames
        if frame_end > total_frames:
            frame_end = total_frames

        tile_size = gif.width // 3

        temp_output_dir = os.path.join(output_dir, "__temp_tiles")
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
        os.makedirs(temp_output_dir)

        for row in range(3):
            for col in range(3):
                os.makedirs(os.path.join(temp_output_dir, f"tile-{row}-{col}"), exist_ok=True)

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

                    folder = os.path.join(temp_output_dir, f"tile-{row}-{col}")
                    tile.save(os.path.join(folder, f"frame-{frame_number}.png"))

            update_progress((frame_number - frame_start + 1) / (frame_end - frame_start) * 100)

        zip_path = os.path.join(output_dir, "gif_tiles_export.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_output_dir):
                for file in files:
                    if file.endswith(".png"):
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, temp_output_dir)
                        zipf.write(abs_path, rel_path)

        shutil.rmtree(temp_output_dir)

        update_status(f"✅ Done! Output zipped to: {zip_path}")
        if not zip_only:
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
            try:
                gif = Image.open(filepath)
                end_frame_entry.delete(0, "end")
                end_frame_entry.insert(0, str(gif.n_frames))
            except:
                pass

    def show_preview(gif_path):
        try:
            gif = Image.open(gif_path)
            gif.seek(0)
            frame = gif.convert("RGBA").resize((120, 120))
            frame_img = ImageTk.PhotoImage(frame)
            preview_label.configure(image=frame_img)
            preview_label.image = frame_img
        except Exception as e:
            status_label.configure(text=f"❌ Preview failed: {e}")

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
        zip_only = zip_only_var.get()

        status_label.configure(text="⏳ Processing...")
        progress_bar.set(0)

        thread = threading.Thread(target=gif2grid_zip, args=(
            gif_path, start_frame, end_frame, output_dir, filter_type,
            lambda msg: app.after(0, status_label.configure, {"text": msg}),
            lambda val: app.after(0, progress_bar.set, val),
            zip_only
        ))
        thread.start()

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")


    global app
    app = ctk.CTk()
    app.title("GIF to Grid Splitter")
    app.geometry("580x600")
    app.resizable(False, False)

    global gif_path_var, output_dir_var, start_frame_entry, end_frame_entry, filter_menu, progress_bar, status_label, preview_label, zip_only_var

    gif_path_var = ctk.StringVar()
    output_dir_var = ctk.StringVar()
    zip_only_var = tk.BooleanVar(value=False)

    ctk.CTkLabel(app, text=" GIF to Grid Splitter", font=("Arial", 22, "bold")).pack(pady=15)

    ctk.CTkButton(app, text="📂 Select GIF", command=select_gif).pack(pady=4)
    ctk.CTkButton(app, text="📁 Select Output Folder", command=select_output_dir).pack(pady=4)

    def change_appearance_mode(new_mode):
        ctk.set_appearance_mode(new_mode)

    ctk.CTkLabel(app, text="🌓 Appearance Mode:").pack()
    appearance_menu = ctk.CTkOptionMenu(app, values=["System", "Light", "Dark" ], command=change_appearance_mode)
    appearance_menu.set("System")
    appearance_menu.pack(pady=(0, 10))





    preview_label = ctk.CTkLabel(app, text="")
    preview_label.pack(pady=8)

    for label_text, var in [("Start Frame:", "0"), ("End Frame:", "100")]:
        ctk.CTkLabel(app, text=label_text).pack()
        entry = ctk.CTkEntry(app, width=120, justify="center")
        entry.insert(0, var)
        entry.pack()
        if "Start" in label_text:
            start_frame_entry = entry
        else:
            end_frame_entry = entry

    ctk.CTkLabel(app, text="Filter:").pack(pady=5)
    filter_menu = ctk.CTkOptionMenu(app, values=["None", "Grayscale", "Tint"])
    filter_menu.pack()

    ctk.CTkCheckBox(app, text="📦 Export as ZIP only (don’t open folder)", variable=zip_only_var).pack(pady=5)

    ctk.CTkButton(app, text="▶️ Start Splitting", command=start_processing).pack(pady=15)

    progress_bar = ctk.CTkProgressBar(app, width=420)
    progress_bar.set(0)
    progress_bar.pack(pady=5)

    status_label = ctk.CTkLabel(app, text="Select a GIF and output folder.", wraplength=500, justify="center")
    status_label.pack(pady=5)

    # dont remove :)
    # shows the contributors of the repo

    def open_contributors():
        webbrowser.open_new("https://github.com/gaxolotl/g2f/graphs/contributors")

    label = ctk.CTkLabel(
        app,
        text="g2f Contributors",
        font=("Arial", 11, "underline"),
        text_color="#ADD8E6",
        cursor="hand2"
    )
    label.pack(side="bottom", pady=8)
    label.bind("<Button-1>", lambda e: open_contributors())

    app.mainloop()

if __name__ == "__main__":
    run_gui()
