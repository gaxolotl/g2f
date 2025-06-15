import os
import zipfile
import shutil
import threading
from PIL import Image, ImageTk, ImageOps
import customtkinter as ctk
from tkinter import filedialog
import tkinter as tk
import webbrowser
import re
import datetime

# This library needs to be installed: pip install rblx-open-cloud
try:
    from rblxopencloud import User, Group, AssetType, ApiKey
except ImportError:
    # Handle case where the library isn't installed
    print("WARNING: rblx-open-cloud is not installed. The uploader will not work.")
    print("Please run: pip install rblx-open-cloud")
    User, Group, AssetType, ApiKey = None, None, None, None


def gif2grid_zip(
    gif_path, frame_start, frame_end, output_dir, apply_filter, update_status,
    update_progress, zip_only=False, rows=3, cols=3, auto_mode=False,
    selected_tile=None
):
    """
    Processes a GIF by splitting its frames into a grid of tiles, applying optional
    filters, and exporting the results into a single ZIP file.
    """
    MAX_AUTO_TILES = 6
    temp_output_dir = os.path.join(output_dir, "__temp_tiles")

    try:
        update_status("⏳ Opening GIF...")
        gif = Image.open(gif_path)
        total_frames = gif.n_frames

        if frame_end > total_frames:
            frame_end = total_frames

        width, height = gif.size

        if auto_mode:
            update_status(f"🤖 Calculating auto grid (max {MAX_AUTO_TILES} tiles)...")
            if height == 0 or width == 0:
                raise ValueError("Image dimensions cannot be zero.")
            
            aspect_ratio = width / height
            best_grid = (1, 1)
            max_found_tiles = 1

            for c in range(1, MAX_AUTO_TILES + 1):
                r_float = c / aspect_ratio
                r = max(1, round(r_float))
                num_tiles = c * r
                
                if num_tiles <= MAX_AUTO_TILES and num_tiles > max_found_tiles:
                    max_found_tiles = num_tiles
                    best_grid = (c, r)

            for r in range(1, MAX_AUTO_TILES + 1):
                c_float = r * aspect_ratio
                c = max(1, round(c_float))
                num_tiles = c * r

                if num_tiles <= MAX_AUTO_TILES and num_tiles > max_found_tiles:
                    max_found_tiles = num_tiles
                    best_grid = (c, r)

            cols, rows = best_grid
             
        tile_width = width // cols
        tile_height = height // rows

        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
        os.makedirs(temp_output_dir)

        tiles_to_process = []
        if selected_tile:
            if selected_tile[0] >= rows or selected_tile[1] >= cols:
                raise ValueError(f"Selected tile {selected_tile} is out of bounds for a {rows}x{cols} grid.")
            tiles_to_process = [selected_tile]
        else:
            tiles_to_process = [(r, c) for r in range(rows) for c in range(cols)]

        for r, c in tiles_to_process:
            os.makedirs(os.path.join(temp_output_dir, f"tile-{r}-{c}"), exist_ok=True)
        
        update_status(f"🖼️ Processing frames for a {rows}x{cols} grid...")
        frames_to_process_count = frame_end - frame_start
        if frames_to_process_count <= 0:
            raise ValueError("Start frame must be less than end frame.")
            
        for i, frame_number in enumerate(range(frame_start, frame_end)):
            gif.seek(frame_number)
            frame = gif.convert("RGBA")

            for row, col in tiles_to_process:
                left = col * tile_width
                upper = row * tile_height
                right = left + tile_width
                lower = upper + tile_height
                tile = frame.crop((left, upper, right, lower))

                if apply_filter == "Grayscale":
                    tile = ImageOps.grayscale(tile).convert("RGBA")
                elif apply_filter == "Tint":
                    r, g, b, a = tile.split()
                    tint_color = (255, 200, 100)
                    r = r.point(lambda x: int(x * (tint_color[0]/255)))
                    g = g.point(lambda x: int(x * (tint_color[1]/255)))
                    b = b.point(lambda x: int(x * (tint_color[2]/255)))
                    tile = Image.merge("RGBA", (r,g,b,a))

                folder = os.path.join(temp_output_dir, f"tile-{row}-{col}")
                tile.save(os.path.join(folder, f"frame-{frame_number}.png"))
            
            update_progress((i + 1) / frames_to_process_count)

        update_status("📦 Zipping files...")
        zip_path = os.path.join(output_dir, "gif_tiles_export.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_output_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, temp_output_dir)
                    zipf.write(abs_path, rel_path)

        shutil.rmtree(temp_output_dir)
        update_status(f"✅ Done! Output zipped to: {os.path.basename(zip_path)}")
        if not zip_only:
            try:
                webbrowser.open(os.path.realpath(output_dir))
            except Exception:
                update_status(f"✅ Done! Could not open folder, find it at: {output_dir}")

    except Exception as e:
        update_status(f"❌ Error: {str(e)}")
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)

# --- Global variable for uploader files ---
selected_upload_files = []

def natural_sort_key(s):
    """
    Create a key for sorting strings in a natural order (e.g., 'item2' before 'item10').
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

def run_gui():
    # --- UI Setup ---
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("Multi-Tool: GIF Splitter & Roblox Uploader")
    app.geometry("600x750") # Increased height for new widget
    app.resizable(False, False)

    ctk.CTkLabel(app, text="Multi-Tool", font=("Arial", 24, "bold")).pack(pady=(10, 5))

    tabview = ctk.CTkTabview(app, width=580)
    tabview.pack(padx=20, pady=5, fill="both", expand=True)

    splitter_tab = tabview.add("GIF Splitter")
    uploader_tab = tabview.add("Roblox Uploader")
    
    # ===================================================================================
    # --- GIF SPLITTER TAB
    # ===================================================================================
    
    # Create a scrollable frame to hold all the controls for this tab
    scrollable_splitter_frame = ctk.CTkScrollableFrame(splitter_tab)
    scrollable_splitter_frame.pack(fill="both", expand=True)
    
    def toggle_grid_entries():
        state = "disabled" if auto_mode_var.get() else "normal"
        rows_entry.configure(state=state)
        cols_entry.configure(state=state)

    def select_gif():
        filepath = filedialog.askopenfilename(filetypes=[("GIF files", "*.gif")])
        if filepath:
            gif_path_var.set(filepath)
            status_label.configure(text=f"📁 Selected: {os.path.basename(filepath)}")
            show_preview(filepath)
            try:
                with Image.open(filepath) as gif:
                    start_frame_entry.delete(0, "end")
                    start_frame_entry.insert(0, "0")
                    end_frame_entry.delete(0, "end")
                    end_frame_entry.insert(0, str(gif.n_frames))
            except Exception as e:
                status_label.configure(text=f"❌ Error reading GIF properties: {e}")

    def show_preview(gif_path):
        try:
            with Image.open(gif_path) as gif:
                gif.seek(0)
                frame = gif.convert("RGBA")
                frame.thumbnail((120, 120))
                frame_img = ImageTk.PhotoImage(frame)
                preview_label.configure(image=frame_img, text="")
                preview_label.image = frame_img
        except Exception as e:
            status_label.configure(text=f"❌ Preview failed: {e}")

    def select_output_dir():
        path = filedialog.askdirectory()
        if path:
            output_dir_var.set(path)
            status_label.configure(text=f"Outputting to: {path}")

    def start_processing():
        gif_path, output_dir = gif_path_var.get(), output_dir_var.get()
        if not gif_path or not output_dir:
            status_label.configure(text="❌ Select both a GIF and an output folder.")
            return

        try:
            start_frame, end_frame = int(start_frame_entry.get()), int(end_frame_entry.get())
            if start_frame < 0 or end_frame <= start_frame: raise ValueError("Frame range is invalid.")
        except ValueError:
            status_label.configure(text="❌ Invalid frame range. Must be numbers.")
            return

        rows, cols, auto_mode = 3, 3, auto_mode_var.get()
        if not auto_mode:
            try:
                rows, cols = int(rows_entry.get()), int(cols_entry.get())
                if rows <= 0 or cols <= 0: raise ValueError
            except ValueError:
                status_label.configure(text="❌ Rows/columns must be positive integers.")
                return

        selected_tile = None
        if tile_text := single_tile_entry.get().strip():
            try:
                r, c = map(int, tile_text.split(","))
                selected_tile = (r, c)
            except (ValueError, IndexError):
                status_label.configure(text="❌ Invalid tile format. Use row,col (e.g., 0,1)")
                return

        status_label.configure(text="⏳ Processing...")
        progress_bar.set(0)
        start_button.configure(state="disabled")

        def update_gui(widget, text_or_val):
            try:
                widget.set(text_or_val)  # Works for progress bar
            except AttributeError:
                widget.configure(text=text_or_val)  # Works for labels
        
        def on_task_done(msg):
            update_gui(status_label, msg)
            start_button.configure(state="normal")

        threading.Thread(target=gif2grid_zip, args=(
            gif_path, start_frame, end_frame, output_dir, filter_menu.get(),
            lambda msg: app.after(0, on_task_done, msg),
            lambda val: app.after(0, update_gui, progress_bar, val),
            zip_only_var.get(), rows, cols, auto_mode, selected_tile
        )).start()

    gif_path_var = ctk.StringVar()
    output_dir_var = ctk.StringVar()
    zip_only_var = ctk.BooleanVar(value=False)
    auto_mode_var = ctk.BooleanVar(value=False)
    
    ctk.CTkButton(scrollable_splitter_frame, text="📂 Select GIF", command=select_gif).pack(pady=5, padx=10, fill="x")
    ctk.CTkButton(scrollable_splitter_frame, text="📁 Select Output Folder", command=select_output_dir).pack(pady=5, padx=10, fill="x")
    preview_label = ctk.CTkLabel(scrollable_splitter_frame, text="GIF Preview will appear here")
    preview_label.pack(pady=10)

    frame_range_frame = ctk.CTkFrame(scrollable_splitter_frame)
    frame_range_frame.pack(pady=5, padx=10, fill="x")
    frame_range_frame.columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(frame_range_frame, text="Start Frame:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
    start_frame_entry = ctk.CTkEntry(frame_range_frame, justify="center")
    start_frame_entry.grid(row=1, column=0, padx=5, pady=2, sticky="ew")
    ctk.CTkLabel(frame_range_frame, text="End Frame:").grid(row=0, column=1, padx=5, pady=2, sticky="w")
    end_frame_entry = ctk.CTkEntry(frame_range_frame, justify="center")
    end_frame_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

    ctk.CTkLabel(scrollable_splitter_frame, text="Filter:").pack(pady=(10, 2))
    filter_menu = ctk.CTkOptionMenu(scrollable_splitter_frame, values=["None", "Grayscale", "Tint"])
    filter_menu.pack()

    ctk.CTkCheckBox(scrollable_splitter_frame, text="📦 Export as ZIP only", variable=zip_only_var).pack(pady=10)
    ctk.CTkLabel(scrollable_splitter_frame, text="Grid Settings:", font=("Arial", 14, "bold")).pack()
    ctk.CTkCheckBox(scrollable_splitter_frame, text="🧠 Auto Grid", variable=auto_mode_var, command=toggle_grid_entries).pack(pady=5)
    
    grid_frame = ctk.CTkFrame(scrollable_splitter_frame)
    grid_frame.pack(pady=5, padx=10, fill="x")
    grid_frame.columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(grid_frame, text="Rows:").grid(row=0, column=0)
    rows_entry = ctk.CTkEntry(grid_frame, width=80, justify="center")
    rows_entry.insert(0, "3")
    rows_entry.grid(row=1, column=0)
    ctk.CTkLabel(grid_frame, text="Columns:").grid(row=0, column=1)
    cols_entry = ctk.CTkEntry(grid_frame, width=80, justify="center")
    cols_entry.insert(0, "3")
    cols_entry.grid(row=1, column=1)

    ctk.CTkLabel(scrollable_splitter_frame, text="Optional: Only Export One Tile").pack(pady=(10, 0))
    single_tile_entry = ctk.CTkEntry(scrollable_splitter_frame, width=140, placeholder_text="row,col (e.g. 0,1)")
    single_tile_entry.pack()
    toggle_grid_entries()

    start_button = ctk.CTkButton(scrollable_splitter_frame, text="▶️ Start Splitting", command=start_processing, height=40)
    start_button.pack(pady=20, padx=10, fill="x")
    progress_bar = ctk.CTkProgressBar(scrollable_splitter_frame)
    progress_bar.set(0)
    progress_bar.pack(pady=5, padx=10, fill="x")
    status_label = ctk.CTkLabel(scrollable_splitter_frame, text="Select a GIF and output folder to begin.", wraplength=500)
    status_label.pack(pady=(5, 10))

    # ===================================================================================
    # --- ROBLOX UPLOADER TAB
    # ===================================================================================
    
    scrollable_uploader_frame = ctk.CTkScrollableFrame(uploader_tab)
    scrollable_uploader_frame.pack(fill="both", expand=True)

    log_file_path_var = ctk.StringVar()

    def select_upload_files():
        global selected_upload_files
        # askopenfilenames returns a tuple, convert to list for sorting
        files = list(filedialog.askopenfilenames(
            title="Select files to upload",
            filetypes=[("Assets", "*.png *.jpg *.mp3 *.ogg *.fbx"), ("All files", "*.*")]
        ))
        if files:
            # Sort files naturally to handle frame_1, frame_2, ..., frame_10 correctly
            files.sort(key=natural_sort_key)
            selected_upload_files = files
            upload_status.configure(text=f"{len(selected_upload_files)} file(s) selected and sorted.")

    def select_log_file():
        filepath = filedialog.asksaveasfilename(
            title="Save Asset ID Log",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if filepath:
            log_file_path_var.set(filepath)
            upload_status.configure(text=f"Log will be saved to: {os.path.basename(filepath)}")

    def start_upload():
        if not all([User, Group, AssetType, ApiKey]):
            upload_status.configure(text="❌ rblx-open-cloud library not found.")
            return

        key, owner_type, owner_id_str = api_key_entry.get().strip(), owner_var.get(), id_entry.get().strip()
        
        if not key or not selected_upload_files or not owner_id_str:
            upload_status.configure(text="❌ API Key, Owner ID, and files are required.")
            return
        
        try:
            owner_id = int(owner_id_str)
        except ValueError:
            upload_status.configure(text="❌ Owner ID must be a number.")
            return
        
        start_upload_button.configure(state="disabled")

        def task():
            app.after(0, lambda: upload_status.configure(text="⏳ Uploading..."))
            app.after(0, lambda: upload_progress.set(0))
            
            try:
                creator = Group(owner_id, api_key=key) if owner_type == "Group" else User(owner_id, api_key=key)
            except Exception as e:
                def update_on_error():
                    upload_status.configure(text=f"❌ Auth Error: {e}")
                    start_upload_button.configure(state="normal")
                app.after(0, update_on_error)
                return

            total = len(selected_upload_files)
            successes = 0
            upload_results = []

            for i, path in enumerate(selected_upload_files):
                ext = os.path.splitext(path)[1].lower()
                atype = (AssetType.Decal if ext in ['.png', '.jpg', '.jpeg', '.gif', '.tga', '.bmp'] else
                         AssetType.Audio if ext in ['.mp3', '.ogg', '.wav', '.flac'] else
                         AssetType.Model if ext == '.fbx' else None)
                if not atype: 
                    print(f"Warning: Skipping unsupported file type: {path}")
                    continue

                try:
                    with open(path, "rb") as file:
                        name = os.path.splitext(os.path.basename(path))[0]
                        asset = creator.upload_asset(file, atype, name, "Uploaded via Multi-Tool.").wait()
                        successes += 1
                        upload_results.append(f"{os.path.basename(path)},{asset.id}")
                        print(f"Uploaded {name} as Asset ID: {asset.id}")
                except Exception as e:
                    print(f"Error uploading {path}: {e}")
                
                app.after(0, upload_progress.set, (i + 1) / total)

            # --- Save log file if a path is provided ---
            log_save_status = ""
            if log_path := log_file_path_var.get():
                try:
                    with open(log_path, "w") as log_file:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_file.write(f"# Roblox Asset Upload Log\n")
                        log_file.write(f"# Generated: {timestamp}\n")
                        log_file.write("Filename,AssetID\n")
                        for result_line in upload_results:
                            log_file.write(f"{result_line}\n")
                    log_save_status = f"\nLog saved to {os.path.basename(log_path)}."
                except Exception as e:
                    log_save_status = f"\n❌ Error saving log: {e}"
                    print(f"Error writing to log file '{log_path}': {e}")
            
            def update_on_finish():
                final_msg = f"✅ Done: {successes}/{total} uploaded."
                if successes != total:
                    final_msg += " (Check console for errors)"
                final_msg += log_save_status # Append the log status message
                upload_status.configure(text=final_msg)
                start_upload_button.configure(state="normal")

            app.after(0, update_on_finish)

        threading.Thread(target=task, daemon=True).start()
    
    ctk.CTkLabel(scrollable_uploader_frame, text="Roblox API Key:", font=("Arial", 14)).pack(pady=(10,2), padx=10, anchor="w")
    api_key_entry = ctk.CTkEntry(scrollable_uploader_frame, placeholder_text="Enter your Open Cloud API Key", show="*")
    api_key_entry.pack(fill="x", padx=10)

    ctk.CTkLabel(scrollable_uploader_frame, text="Owner (Upload Destination):", font=("Arial", 14)).pack(pady=(10,2), padx=10, anchor="w")
    owner_frame = ctk.CTkFrame(scrollable_uploader_frame)
    owner_frame.pack(fill="x", padx=10)
    owner_var = ctk.StringVar(value="Group")
    owner_type_menu = ctk.CTkSegmentedButton(owner_frame, values=["User", "Group"], variable=owner_var)
    owner_type_menu.pack(side="left", padx=(0,10))
    id_entry = ctk.CTkEntry(owner_frame, placeholder_text="Enter User ID or Group ID")
    id_entry.pack(fill="x", expand=True)

    ctk.CTkButton(scrollable_uploader_frame, text="📂 Select Files to Upload", command=select_upload_files).pack(pady=20, padx=10, fill="x")

    # --- New Log File Widgets ---
    ctk.CTkLabel(scrollable_uploader_frame, text="Optional: Save Asset IDs to Log File", font=("Arial", 14)).pack(pady=(10,2), padx=10, anchor="w")
    log_file_frame = ctk.CTkFrame(scrollable_uploader_frame)
    log_file_frame.pack(fill="x", padx=10, pady=(0, 10))
    log_file_button = ctk.CTkButton(log_file_frame, text="Set Log File", command=select_log_file, width=120)
    log_file_button.pack(side="left", padx=(0, 10))
    log_file_entry = ctk.CTkEntry(log_file_frame, textvariable=log_file_path_var, placeholder_text="No log file selected", state="disabled")
    log_file_entry.pack(side="left", fill="x", expand=True)

    start_upload_button = ctk.CTkButton(scrollable_uploader_frame, text="🚀 Start Upload", command=start_upload, height=40)
    start_upload_button.pack(pady=10, padx=10, fill="x")

    upload_progress = ctk.CTkProgressBar(scrollable_uploader_frame)
    upload_progress.set(0)
    upload_progress.pack(pady=5, padx=10, fill="x")

    upload_status = ctk.CTkLabel(scrollable_uploader_frame, text="Enter API info and select files.", wraplength=500)
    upload_status.pack(pady=10)

    app.mainloop()

if __name__ == "__main__":
    run_gui()