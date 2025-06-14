import os
import zipfile
import shutil
import threading
from PIL import Image, ImageTk, ImageEnhance, ImageOps
import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
import webbrowser
from math import gcd
import rblxopencloud

def upload_tiles_to_roblox(tile_dir, user_id, api_key, update_status):
    """Upload tiles to Roblox as decals."""
    try:
        user = rblxopencloud.User(user_id=user_id, api_key=api_key)
        update_status("☁️ Connecting to Roblox...")

        uploaded_count = 0
        for root, _, files in os.walk(tile_dir):
            for file in files:
                if file.lower().endswith(".png"):
                    try:
                        with open(os.path.join(root, file), "rb") as f:
                            asset = user.upload_asset(
                                f,
                                rblxopencloud.AssetType.Decal,
                                name=file,
                                description="Uploaded via GIF to Grid Splitter"
                            ).wait()
                        uploaded_count += 1
                        update_status(f"✅ Uploaded {file} (ID: {asset.id})")
                    except Exception as e:
                        update_status(f"⚠️ Failed to upload {file}: {e}")

        update_status(f"✅ Uploaded {uploaded_count} tiles to Roblox.")
    except Exception as e:
        update_status(f"❌ Roblox upload failed: {str(e)}")

def gif2grid_zip(
    gif_path, frame_start, frame_end, output_dir, apply_filter, update_status,
    update_progress, zip_only=False, rows=3, cols=3, auto_mode=False,
    selected_tile=None, upload_to_roblox=False, roblox_user_id=None, roblox_api_key=None
):
    # hello i recommend that you do not change this because
    # from what i know this is the best size
    MAX_AUTO_TILES = 6
    
    try:
        # Validate input file exists and is readable
        if not os.path.exists(gif_path):
            raise FileNotFoundError(f"GIF file not found: {gif_path}")
        
        update_status("⏳ Opening GIF...")
        gif = Image.open(gif_path)
        
        # Validate GIF format
        if not gif.is_animated:
            raise ValueError("Selected file is not an animated GIF")
            
        total_frames = gif.n_frames

        # Validate frame range
        if frame_start < 0:
            frame_start = 0
        if frame_end > total_frames:
            frame_end = total_frames
        if frame_start >= frame_end:
            raise ValueError("Start frame must be less than end frame")

        width, height = gif.size

        # Validate image dimensions
        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions")

        if auto_mode:
            update_status(f"🤖 Calculating auto grid (max {MAX_AUTO_TILES} tiles)...")
            if height == 0 or width == 0:
                raise ValueError("Image dimensions cannot be zero.")
            
            aspect_ratio = width / height
            
            best_grid = (1, 1) # Default to 1x1
            max_found_tiles = 1

            # Iterate through possible columns to find the best fit
            for c in range(1, MAX_AUTO_TILES + 1):
                # Calculate ideal rows for this many columns to match aspect ratio
                r_float = c / aspect_ratio
                r = max(1, round(r_float)) # Ensure at least 1 row
                
                num_tiles = c * r
                
                # If it's a valid grid (under the limit) and better than what we have, save it
                if num_tiles <= MAX_AUTO_TILES and num_tiles > max_found_tiles:
                    max_found_tiles = num_tiles
                    best_grid = (c, r)

            # Also check the inverse (iterating rows) for tall/thin images
            for r in range(1, MAX_AUTO_TILES + 1):
                c_float = r * aspect_ratio
                c = max(1, round(c_float))

                num_tiles = c * r
                if num_tiles <= MAX_AUTO_TILES and num_tiles > max_found_tiles:
                    max_found_tiles = num_tiles
                    best_grid = (c, r)

            cols, rows = best_grid

        # Validate grid dimensions
        if rows <= 0 or cols <= 0:
            raise ValueError("Grid dimensions must be positive")
            
        tile_width = width // cols
        tile_height = height // rows

        # Ensure minimum tile size
        if tile_width < 16 or tile_height < 16:
            raise ValueError("Grid division would create tiles that are too small (minimum 16x16 pixels)")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        temp_output_dir = os.path.join(output_dir, "__temp_tiles")
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
        os.makedirs(temp_output_dir)

        # Determine which tiles to process
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
            try:
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
                        tint_color = (255, 200, 100) # Example: Sepia-like tint
                        r = r.point(lambda x: int(x * (tint_color[0]/255)))
                        g = g.point(lambda x: int(x * (tint_color[1]/255)))
                        b = b.point(lambda x: int(x * (tint_color[2]/255)))
                        tile = Image.merge("RGBA", (r,g,b,a))

                    folder = os.path.join(temp_output_dir, f"tile-{row}-{col}")
                    tile.save(os.path.join(folder, f"frame-{frame_number}.png"))
                
                update_progress((i + 1) / frames_to_process_count)
            except Exception as e:
                update_status(f"⚠️ Error processing frame {frame_number}: {str(e)}")
                continue

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
        
        # Handle Roblox upload if enabled
        if upload_to_roblox and roblox_user_id and roblox_api_key:
            try:
                upload_tiles_to_roblox(temp_output_dir, roblox_user_id, roblox_api_key, update_status)
            except Exception as e:
                update_status(f"❌ Roblox upload failed: {str(e)}")
        
        if not zip_only:
            try:
                webbrowser.open(f'file:///{os.path.realpath(output_dir)}')
            except Exception:
                update_status(f"✅ Done! Could not open folder, find it at: {output_dir}")

    except Exception as e:
        update_status(f"❌ Error: {str(e)}")
        if 'temp_output_dir' in locals() and os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
        raise  # Re-raise the exception to be handled by the GUI

def run_gui():
    # --- UI Helper Functions ---
    def toggle_grid_entries():
        """Enable/disable manual row/col entries based on auto-mode checkbox."""
        if auto_mode_var.get():
            rows_entry.configure(state="disabled")
            cols_entry.configure(state="disabled")
        else:
            rows_entry.configure(state="normal")
            cols_entry.configure(state="normal")

    def validate_frame_range():
        """Validate frame range entries."""
        try:
            start = int(start_frame_entry.get())
            end = int(end_frame_entry.get())
            if start < 0:
                start_frame_entry.delete(0, "end")
                start_frame_entry.insert(0, "0")
            if end <= start:
                end_frame_entry.delete(0, "end")
                end_frame_entry.insert(0, str(start + 1))
        except ValueError:
            pass

    def validate_grid_entries():
        """Validate grid dimension entries."""
        try:
            rows = int(rows_entry.get())
            cols = int(cols_entry.get())
            if rows < 1:
                rows_entry.delete(0, "end")
                rows_entry.insert(0, "1")
            if cols < 1:
                cols_entry.delete(0, "end")
                cols_entry.insert(0, "1")
        except ValueError:
            pass

    # --- Core GUI Functions ---
    def select_gif():
        filepath = filedialog.askopenfilename(filetypes=[("GIF files", "*.gif")])
        if filepath:
            try:
                # Validate GIF before accepting
                with Image.open(filepath) as test_gif:
                    if not test_gif.is_animated:
                        status_label.configure(text="❌ Selected file is not an animated GIF")
                        return
                    # Set default frame range based on GIF
                    start_frame_entry.delete(0, "end")
                    start_frame_entry.insert(0, "0")
                    end_frame_entry.delete(0, "end")
                    end_frame_entry.insert(0, str(test_gif.n_frames))
                
                gif_path_var.set(filepath)
                status_label.configure(text=f"📁 Selected: {os.path.basename(filepath)}")
                show_preview(filepath)
            except Exception as e:
                status_label.configure(text=f"❌ Error reading GIF: {str(e)}")

    def show_preview(gif_path):
        try:
            with Image.open(gif_path) as gif:
                gif.seek(0)
                # Maintain aspect ratio for preview
                frame = gif.convert("RGBA")
                frame.thumbnail((120, 120))
                frame_img = ImageTk.PhotoImage(frame)
                preview_label.configure(image=frame_img, text="")
                preview_label.image = frame_img # Keep a reference
        except Exception as e:
            status_label.configure(text=f"❌ Preview failed: {str(e)}")
            preview_label.configure(image=None, text="Preview failed")

    def select_output_dir():
        path = filedialog.askdirectory()
        if path:
            try:
                # Test write permissions
                test_file = os.path.join(path, ".write_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                output_dir_var.set(path)
            except Exception as e:
                status_label.configure(text=f"❌ Cannot write to selected directory: {str(e)}")

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
                raise ValueError("Frame range is invalid.")
        except ValueError as e:
            status_label.configure(text=f"❌ Invalid frame range. {e}")
            return

        rows, cols = 3, 3 # Default values
        auto_mode = auto_mode_var.get()
        if not auto_mode:
            try:
                rows = int(rows_entry.get())
                cols = int(cols_entry.get())
                if rows <= 0 or cols <= 0:
                    raise ValueError
            except ValueError:
                status_label.configure(text="❌ Rows/columns must be positive integers.")
                return

        tile_text = single_tile_entry.get().strip()
        selected_tile = None
        if tile_text:
            try:
                r, c = map(int, tile_text.split(","))
                selected_tile = (r, c)
            except (ValueError, IndexError):
                status_label.configure(text="❌ Invalid tile format. Use row,col (e.g., 0,1)")
                return

        filter_type = filter_menu.get()
        zip_only = zip_only_var.get()
        
        # Get Roblox settings
        upload_to_roblox = roblox_upload_var.get()
        roblox_user_id = roblox_user_id_var.get().strip()
        roblox_api_key = roblox_api_key_var.get().strip()
        
        if upload_to_roblox and (not roblox_user_id or not roblox_api_key):
            status_label.configure(text="❌ Please provide both Roblox User ID and API Key for upload.")
            return

        status_label.configure(text="⏳ Processing...")
        progress_bar.set(0)
        start_button.configure(state="disabled") # Disable button during processing

        def on_task_done(msg):
            status_label.configure(text=msg)
            start_button.configure(state="normal") # Re-enable button

        thread = threading.Thread(target=gif2grid_zip, args=(
            gif_path, start_frame, end_frame, output_dir, filter_type,
            lambda msg: app.after(0, on_task_done, msg), # Pass final message to on_task_done
            lambda val: app.after(0, progress_bar.set, val),
            zip_only, rows, cols, auto_mode, selected_tile,
            upload_to_roblox, roblox_user_id, roblox_api_key
        ))
        thread.daemon = True  # Make thread daemon so it exits when main program exits
        thread.start()

    # --- GUI Setup ---
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("GIF to Grid Splitter")
    app.geometry("580x650")
    app.resizable(False, False)

    # --- Place widgets into a scrollable frame ---
    ctk.CTkLabel(app, text=" GIF to Grid Splitter", font=("Arial", 24, "bold")).pack(pady=(10, 5))
    
    scrollable_frame = ctk.CTkScrollableFrame(app, label_text="Controls")
    scrollable_frame.pack(fill="both", expand=True, padx=15, pady=10)

    # Variables
    gif_path_var = ctk.StringVar()
    output_dir_var = ctk.StringVar()
    zip_only_var = tk.BooleanVar(value=False)
    auto_mode_var = tk.BooleanVar(value=False)
    roblox_user_id_var = ctk.StringVar()
    roblox_api_key_var = ctk.StringVar()
    roblox_upload_var = tk.BooleanVar(value=False)

    # --- File Selection Widgets ---
    ctk.CTkButton(scrollable_frame, text="📂 Select GIF", command=select_gif).pack(pady=5, fill="x", expand=True)
    ctk.CTkButton(scrollable_frame, text="📁 Select Output Folder", command=select_output_dir).pack(pady=5, fill="x", expand=True)

    preview_label = ctk.CTkLabel(scrollable_frame, text="GIF Preview will appear here")
    preview_label.pack(pady=10)

    # --- Frame Range Widgets ---
    frame_range_frame = ctk.CTkFrame(scrollable_frame)
    frame_range_frame.pack(pady=5, fill="x", expand=True)
    frame_range_frame.columnconfigure((0, 1), weight=1)

    ctk.CTkLabel(frame_range_frame, text="Start Frame:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
    start_frame_entry = ctk.CTkEntry(frame_range_frame, width=120, justify="center")
    start_frame_entry.insert(0, "0")
    start_frame_entry.grid(row=1, column=0, padx=5, pady=2)
    start_frame_entry.bind("<FocusOut>", lambda e: validate_frame_range())

    ctk.CTkLabel(frame_range_frame, text="End Frame:").grid(row=0, column=1, padx=5, pady=2, sticky="w")
    end_frame_entry = ctk.CTkEntry(frame_range_frame, width=120, justify="center")
    end_frame_entry.insert(0, "100")
    end_frame_entry.grid(row=1, column=1, padx=5, pady=2)
    end_frame_entry.bind("<FocusOut>", lambda e: validate_frame_range())

    # --- Filter & Options Widgets ---
    ctk.CTkLabel(scrollable_frame, text="Filter:").pack(pady=(10, 2))
    filter_menu = ctk.CTkOptionMenu(scrollable_frame, values=["None", "Grayscale", "Tint"])
    filter_menu.set("None")
    filter_menu.pack()

    ctk.CTkCheckBox(scrollable_frame, text="📦 Export as ZIP only (don't open folder)", variable=zip_only_var).pack(pady=10)
    
    # --- Roblox Upload Section ---
    ctk.CTkLabel(scrollable_frame, text="Roblox Upload (Optional):", font=("Arial", 14, "bold")).pack(pady=(10, 5))
    ctk.CTkCheckBox(scrollable_frame, text="☁️ Upload tiles as Roblox decals", variable=roblox_upload_var).pack(pady=5)

    roblox_frame = ctk.CTkFrame(scrollable_frame)
    roblox_frame.pack(pady=5, fill="x", expand=True)

    ctk.CTkLabel(roblox_frame, text="User ID:").grid(row=0, column=0, padx=5, sticky="w")
    ctk.CTkEntry(roblox_frame, textvariable=roblox_user_id_var).grid(row=0, column=1, padx=5, sticky="ew")

    ctk.CTkLabel(roblox_frame, text="API Key:").grid(row=1, column=0, padx=5, sticky="w")
    ctk.CTkEntry(roblox_frame, textvariable=roblox_api_key_var, show="*").grid(row=1, column=1, padx=5, sticky="ew")

    roblox_frame.columnconfigure(1, weight=1)

    # --- Grid Settings Widgets ---
    ctk.CTkLabel(scrollable_frame, text="Grid Settings:", font=("Arial", 14, "bold")).pack(pady=(10, 5))
    ctk.CTkCheckBox(scrollable_frame, text="🧠 Auto Grid (best fit based on aspect ratio)", variable=auto_mode_var, command=toggle_grid_entries).pack(pady=5)

    grid_frame = ctk.CTkFrame(scrollable_frame)
    grid_frame.pack(pady=5, fill="x", expand=True)
    grid_frame.columnconfigure((0, 1), weight=1)

    ctk.CTkLabel(grid_frame, text="Rows:").grid(row=0, column=0)
    rows_entry = ctk.CTkEntry(grid_frame, width=80, justify="center")
    rows_entry.insert(0, "3")
    rows_entry.grid(row=1, column=0)
    rows_entry.bind("<FocusOut>", lambda e: validate_grid_entries())

    ctk.CTkLabel(grid_frame, text="Columns:").grid(row=0, column=1)
    cols_entry = ctk.CTkEntry(grid_frame, width=80, justify="center")
    cols_entry.insert(0, "3")
    cols_entry.grid(row=1, column=1)
    cols_entry.bind("<FocusOut>", lambda e: validate_grid_entries())

    ctk.CTkLabel(scrollable_frame, text="Optional: Only Export One Tile (e.g., 0,1 for row 0, col 1)").pack(pady=(10, 0))
    single_tile_entry = ctk.CTkEntry(scrollable_frame, width=140, placeholder_text="row,col")
    single_tile_entry.pack()
    
    toggle_grid_entries() # Set initial state for grid entries

    # --- Action Button & Status Widgets ---
    start_button = ctk.CTkButton(app, text="▶️ Start Splitting", command=start_processing, height=40)
    start_button.pack(pady=10, padx=15, fill="x")

    progress_bar = ctk.CTkProgressBar(app, width=420)
    progress_bar.set(0)
    progress_bar.pack(pady=5, padx=15, fill="x")

    status_label = ctk.CTkLabel(app, text="Select a GIF and output folder to begin.", wraplength=550, justify="center")
    status_label.pack(pady=(5, 10))

    # Handle window close
    def on_closing():
        if start_button.cget("state") == "disabled":
            if messagebox.askokcancel("Quit", "Processing is in progress. Are you sure you want to quit?"):
                app.destroy()
        else:
            app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()

if __name__ == "__main__":
    run_gui()
