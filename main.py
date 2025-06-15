import os
import zipfile
import shutil
import threading
from PIL import Image, ImageTk, ImageOps, ImageEnhance, ImageFilter
import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
import webbrowser
import re
import datetime
import traceback
import sys
import tempfile
import hashlib
import time
from typing import Optional, Tuple, List, Dict, Any, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import importlib.util
import logging
from dataclasses import dataclass
from queue import Queue, PriorityQueue
import gc
from functools import lru_cache, wraps
import numpy as np
import signal
import atexit
import weakref
from abc import ABC, abstractmethod
import asyncio
import aiofiles
import aiohttp
from typing_extensions import TypedDict, Literal
import colorama
from tqdm import tqdm
import cv2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import math

def check_and_install_dependencies():
    """Check and install required dependencies."""
    required_packages = {
        'psutil': 'psutil',
        'rblxopencloud': 'rblxopencloud',
        'PIL': 'Pillow',
        'customtkinter': 'customtkinter',
        'numpy': 'numpy',
        'aiofiles': 'aiofiles',
        'aiohttp': 'aiohttp',
        'colorama': 'colorama',
        'tqdm': 'tqdm',
        'opencv-python': 'opencv-python',
        'watchdog': 'watchdog',
        'typing-extensions': 'typing-extensions'
    }
    
    print("Checking and installing required packages...")
    missing_packages = []
    
    for module, package in required_packages.items():
        try:
            if module == 'PIL':
                importlib.import_module('PIL')
            elif module == 'cv2':
                importlib.import_module('cv2')
            else:
                importlib.import_module(module)
            print(f"✓ {package} is already installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} needs to be installed")
    
    if missing_packages:
        print("\nInstalling missing packages...")
        try:
            # First upgrade pip itself
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Install all missing packages
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + missing_packages)
            print("\nSuccessfully installed all required packages!")
            
            # Force reload modules after installation
            for module in missing_packages:
                try:
                    if module == 'Pillow':
                        importlib.import_module('PIL')
                    elif module == 'opencv-python':
                        importlib.import_module('cv2')
                    else:
                        importlib.import_module(module)
                    print(f"✓ Successfully imported {module}")
                except ImportError as e:
                    print(f"⚠ Warning: Failed to import {module} after installation: {e}")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error installing packages: {e}")
            print("\nPlease try installing the packages manually using:")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)
    else:
        print("\nAll required packages are already installed!")
    
    print("\nDependency check complete!")

# Check and install dependencies before importing them
check_and_install_dependencies()

# Now import all required modules
try:
    from PIL import Image, ImageTk, ImageOps, ImageEnhance, ImageFilter
    import customtkinter as ctk
    from tkinter import filedialog, messagebox
    import tkinter as tk
    import webbrowser
    import re
    import traceback
    import psutil
    import aiofiles
    import aiohttp
    from typing_extensions import TypedDict, Literal
    import colorama
    from tqdm import tqdm
    import cv2
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    import rblxopencloud
    print("Successfully imported all required modules!")
except ImportError as e:
    print(f"\n❌ Error importing required modules: {e}")
    print("Please try running the script again or install the missing packages manually.")
    sys.exit(1)

# This library needs to be installed: pip install rblx-open-cloud
try:
    from rblxopencloud import User, Group, AssetType, ApiKey
except ImportError:
    # Handle case where the library isn't installed
    print("WARNING: rblx-open-cloud is not installed. The uploader will not work.")
    print("Please run: pip install rblx-open-cloud")
    User, Group, AssetType, ApiKey = None, None, None, None

# Constants
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds
MAX_MEMORY_USAGE = 0.8  # 80% of available memory
TEMP_DIR_PREFIX = "gif_splitter_"
BACKUP_EXTENSION = ".backup"
MAX_CACHE_SIZE = 1000
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for file operations
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
MAX_FRAMES = 1000  # Maximum frames allowed
MAX_DIMENSION = 4096  # Maximum dimension allowed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gif_splitter.log'),
        logging.StreamHandler()
    ]
)

# Initialize colorama for cross-platform colored terminal output
colorama.init()

@dataclass
class ProcessingConfig:
    """Configuration for image processing."""
    quality: int = 95
    optimize: bool = True
    compression_level: int = 6
    max_dimension: int = 4096
    chunk_size: int = CHUNK_SIZE

class EnhancedMemoryManager:
    """Enhanced memory management with advanced features."""
    def __init__(self, max_usage: float = MAX_MEMORY_USAGE):
        self.max_usage = max_usage
        self.process = psutil.Process()
        self.memory_history = []
        self.last_gc_time = time.time()
        self.gc_interval = 60  # seconds
    
    def check_memory(self) -> bool:
        """Check if current memory usage is below threshold."""
        memory_percent = self.process.memory_percent()
        self.memory_history.append(memory_percent)
        if len(self.memory_history) > 100:
            self.memory_history.pop(0)
        return memory_percent < (self.max_usage * 100)
    
    def wait_for_memory(self) -> None:
        """Wait until memory usage is below threshold with smart GC."""
        while not self.check_memory():
            current_time = time.time()
            if current_time - self.last_gc_time > self.gc_interval:
                gc.collect()
                self.last_gc_time = current_time
            time.sleep(0.1)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get detailed memory statistics."""
        return {
            'current': self.process.memory_percent(),
            'average': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
            'peak': max(self.memory_history) if self.memory_history else 0
        }

class EnhancedFileValidator:
    """Enhanced file validation with more features."""
    @staticmethod
    def validate_api_key(key: str) -> bool:
        """Validate Roblox API key format with enhanced checks."""
        if not key or len(key) < 32:
            return False
        return bool(re.match(r'^[A-Za-z0-9_-]{32,}$', key))
    
    @staticmethod
    def validate_file_size(file_path: str, max_size: int) -> bool:
        """Validate file size with chunked reading."""
        try:
            size = 0
            with open(file_path, 'rb') as f:
                while chunk := f.read(CHUNK_SIZE):
                    size += len(chunk)
                    if size > max_size:
                        return False
            return True
        except OSError:
            return False
    
    @staticmethod
    def validate_file_type(file_path: str, allowed_extensions: set) -> bool:
        """Validate file extension and content type."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in allowed_extensions:
            return False
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False

class EnhancedBackupManager:
    """Enhanced backup management with versioning."""
    def __init__(self, max_backups: int = 5):
        self.max_backups = max_backups
    
    def create_backup(self, file_path: str) -> str:
        """Create a versioned backup of a file."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.{timestamp}{BACKUP_EXTENSION}"
        try:
            shutil.copy2(file_path, backup_path)
            self._cleanup_old_backups(file_path)
            return backup_path
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
            return None
    
    def _cleanup_old_backups(self, original_path: str) -> None:
        """Clean up old backups keeping only the most recent ones."""
        backup_pattern = f"{original_path}.*{BACKUP_EXTENSION}"
        backups = sorted(Path(original_path).parent.glob(backup_pattern))
        if len(backups) > self.max_backups:
            for old_backup in backups[:-self.max_backups]:
                try:
                    old_backup.unlink()
                except Exception as e:
                    logging.error(f"Failed to delete old backup {old_backup}: {e}")

class EnhancedRetryManager:
    """Enhanced retry management with advanced features."""
    @staticmethod
    def retry_operation(operation, *args, max_attempts: int = MAX_RETRY_ATTEMPTS, 
                       delay: int = RETRY_DELAY, **kwargs) -> Any:
        """Retry an operation with exponential backoff and logging."""
        last_exception = None
        for attempt in range(max_attempts):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_attempts - 1:
                    sleep_time = delay * (2 ** attempt)
                    logging.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        logging.error(f"All {max_attempts} attempts failed")
        raise last_exception

class ImageProcessor:
    """Enhanced image processing with advanced features."""
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.memory_manager = EnhancedMemoryManager()
    
    @lru_cache(maxsize=MAX_CACHE_SIZE)
    def process_image(self, image: Image.Image, filter_type: str) -> Image.Image:
        """Process image with caching and memory management."""
        self.memory_manager.wait_for_memory()
        
        if filter_type == "Grayscale":
            return ImageOps.grayscale(image).convert("RGBA")
        elif filter_type == "Tint":
            r, g, b, a = image.split()
            tint_color = (255, 200, 100)
            r = r.point(lambda x: int(x * (tint_color[0]/255)))
            g = g.point(lambda x: int(x * (tint_color[1]/255)))
            b = b.point(lambda x: int(x * (tint_color[2]/255)))
            return Image.merge("RGBA", (r,g,b,a))
        elif filter_type == "Enhance":
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.5)
        return image

    def save_image(self, image: Image.Image, path: str) -> None:
        """Save image with optimization."""
        try:
            image.save(
                path,
                optimize=self.config.optimize,
                quality=self.config.quality,
                compress_level=self.config.compression_level
            )
        finally:
            image.close()

class ParallelProcessor:
    """Enhanced parallel processing with advanced features."""
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(os.cpu_count(), 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.memory_manager = EnhancedMemoryManager()
    
    def process_frames(self, frames: List[int], process_func: callable) -> List[Any]:
        """Process frames in parallel with enhanced error handling."""
        results = []
        futures = []
        
        for frame in frames:
            future = self.executor.submit(self._safe_process, process_func, frame)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Frame processing failed: {e}")
                results.append(None)
        
        return results
    
    def _safe_process(self, process_func: callable, frame: int) -> Any:
        """Safely process a frame with memory management."""
        self.memory_manager.wait_for_memory()
        return process_func(frame)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)

def validate_gif(file_path: str) -> bool:
    """Validate GIF file with enhanced checks."""
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return False
        
        if not EnhancedFileValidator.validate_file_size(file_path, MAX_FILE_SIZE):
            logging.error(f"File size exceeds limit: {file_path}")
            return False
        
        if not EnhancedFileValidator.validate_file_type(file_path, {'.gif'}):
            logging.error(f"Invalid file type: {file_path}")
            return False
        
        with Image.open(file_path) as img:
            if not img.is_animated:
                logging.error("File is not an animated GIF")
                return False
            
            # Validate frame count
            frame_count = getattr(img, 'n_frames', 1)
            if frame_count > MAX_FRAMES:
                logging.error(f"Too many frames: {frame_count}")
                return False
            
            # Validate frame dimensions
            for frame in range(frame_count):
                img.seek(frame)
                if img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
                    logging.error(f"Frame {frame} exceeds maximum dimensions")
                    return False
        
        return True
    except Exception as e:
        logging.error(f"Error validating GIF: {e}")
        return False

def show_error(title: str, message: str, error_details: Optional[str] = None) -> None:
    """Show error message with optional details."""
    if error_details:
        message += f"\n\nDetails: {error_details}"
    messagebox.showerror(title, message)

def handle_exception(exc_type: type, exc_value: Exception, exc_traceback: Any) -> None:
    """Global exception handler."""
    error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(f"Unhandled exception:\n{error_msg}")
    show_error("Unexpected Error", 
               "An unexpected error occurred. The application will continue running, but you may want to restart it.",
               error_msg)

sys.excepthook = handle_exception

def natural_sort_key(s: str) -> List[Any]:
    """Create a key for sorting strings in a natural order (e.g., 'item2' before 'item10')."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

def gif2grid_zip(gif_path: str, output_path: str, progress_callback: Optional[Callable] = None) -> bool:
    """Convert GIF to grid of frames and create zip file with enhanced features."""
    try:
        if not validate_gif(gif_path):
            return False
        
        memory_manager = EnhancedMemoryManager()
        backup_manager = EnhancedBackupManager()
        
        # Create backup of existing zip if it exists
        if os.path.exists(output_path):
            backup_path = backup_manager.create_backup(output_path)
            if not backup_path:
                logging.error("Failed to create backup of existing zip")
                return False
        
        with Image.open(gif_path) as img:
            frame_count = getattr(img, 'n_frames', 1)
            grid_size = int(math.ceil(math.sqrt(frame_count)))
            
            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                frames = []
                for i in range(frame_count):
                    img.seek(i)
                    frame = img.copy()
                    frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
                    frame.save(frame_path, "PNG", optimize=True)
                    frames.append(frame_path)
                    memory_manager.wait_for_memory()
                
                # Create zip file
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for i, frame_path in enumerate(frames):
                        zipf.write(frame_path, f"frame_{i:03d}.png")
                        if progress_callback:
                            progress_callback((i + 1) / frame_count * 100)
                        memory_manager.wait_for_memory()
        
        return True
    except Exception as e:
        logging.error(f"Error in gif2grid_zip: {e}")
        return False

def start_upload(file_path: str, api_key: str, progress_callback: Optional[Callable] = None) -> None:
    """Start file upload with enhanced error handling and retry mechanism."""
    try:
        if not EnhancedFileValidator.validate_api_key(api_key):
            logging.error("Invalid API key format")
            return
        
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return
        
        def upload_operation():
            try:
                with open(file_path, 'rb') as f:
                    asset = EnhancedRetryManager.retry_operation(
                        lambda: rblxopencloud.upload_asset(
                            api_key=api_key,
                            file=f,
                            asset_type="Decal",
                            name=os.path.basename(file_path)
                        )
                    )
                logging.info(f"Uploaded {os.path.basename(file_path)} as Asset ID: {asset.id}")
                return asset
            except Exception as e:
                logging.error(f"Upload failed: {e}")
                raise
        
        threading.Thread(
            target=lambda: EnhancedRetryManager.retry_operation(
                upload_operation,
                max_attempts=MAX_RETRY_ATTEMPTS,
                delay=RETRY_DELAY
            )
        ).start()
    except Exception as e:
        logging.error(f"Error in start_upload: {e}")

# --- Global variable for uploader files ---
selected_upload_files: List[str] = []

class ThreadManager:
    """Manages thread lifecycle and monitoring."""
    def __init__(self):
        self.active_threads = weakref.WeakSet()
        self.thread_status = {}
        self.thread_results = {}
        self._lock = threading.Lock()
    
    def start_thread(self, target: Callable, args: tuple = (), kwargs: dict = None, 
                    name: str = None, daemon: bool = True) -> threading.Thread:
        """Start a new thread with monitoring."""
        if kwargs is None:
            kwargs = {}
        
        def wrapped_target(*args, **kwargs):
            thread_id = threading.get_ident()
            try:
                with self._lock:
                    self.thread_status[thread_id] = "running"
                result = target(*args, **kwargs)
                with self._lock:
                    self.thread_status[thread_id] = "completed"
                    self.thread_results[thread_id] = result
            except Exception as e:
                with self._lock:
                    self.thread_status[thread_id] = "failed"
                    self.thread_results[thread_id] = e
                logging.error(f"Thread {name or thread_id} failed: {e}")
                raise
        
        thread = threading.Thread(
            target=wrapped_target,
            args=args,
            kwargs=kwargs,
            name=name,
            daemon=daemon
        )
        
        with self._lock:
            self.active_threads.add(thread)
            self.thread_status[thread.ident] = "starting"
        
        thread.start()
        return thread
    
    def get_thread_status(self, thread: threading.Thread) -> str:
        """Get the status of a thread."""
        with self._lock:
            return self.thread_status.get(thread.ident, "unknown")
    
    def get_thread_result(self, thread: threading.Thread) -> Any:
        """Get the result of a completed thread."""
        with self._lock:
            return self.thread_results.get(thread.ident)
    
    def get_active_threads(self) -> List[threading.Thread]:
        """Get list of active threads."""
        with self._lock:
            return list(self.active_threads)
    
    def wait_for_threads(self, timeout: float = None) -> bool:
        """Wait for all active threads to complete."""
        start_time = time.time()
        while True:
            active_threads = self.get_active_threads()
            if not active_threads:
                return True
            
            if timeout is not None and time.time() - start_time > timeout:
                return False
            
            for thread in active_threads:
                thread.join(0.1)
    
    def cleanup(self):
        """Clean up completed threads."""
        with self._lock:
            completed_threads = [
                thread_id for thread_id, status in self.thread_status.items()
                if status in ("completed", "failed")
            ]
            for thread_id in completed_threads:
                del self.thread_status[thread_id]
                if thread_id in self.thread_results:
                    del self.thread_results[thread_id]

# Create a global thread manager
thread_manager = ThreadManager()

def run_gui() -> None:
    """Initialize and run the main GUI application."""
    # --- UI Setup ---
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("GIF Splitter & Roblox Uploader")
    app.geometry("600x750")
    app.resizable(False, False)

    # Add glass effect to main window
    app.configure(fg_color=("gray90", "gray13"))
    app.attributes('-alpha', 0.97)

    # Create a main frame with glass effect
    main_frame = ctk.CTkFrame(app, fg_color=("gray85", "gray15"), corner_radius=15)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Title with glass effect
    title_frame = ctk.CTkFrame(main_frame, fg_color=("gray80", "gray20"), corner_radius=10)
    title_frame.pack(fill="x", padx=20, pady=(10, 5))
    ctk.CTkLabel(title_frame, text="Gif splitter", font=("Arial", 24, "bold")).pack(pady=10)

    # Tabview with glass effect
    tabview = ctk.CTkTabview(main_frame, width=580, fg_color=("gray85", "gray15"), corner_radius=15)
    tabview.pack(padx=20, pady=5, fill="both", expand=True)

    splitter_tab = tabview.add("GIF Splitter")
    uploader_tab = tabview.add("Roblox Uploader")
    
    # ===================================================================================
    # --- GIF SPLITTER TAB
    # ===================================================================================
    
    # Create a scrollable frame with glass effect
    scrollable_splitter_frame = ctk.CTkScrollableFrame(splitter_tab, fg_color=("gray87", "gray13"), corner_radius=10)
    scrollable_splitter_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Style for different widget types
    button_style = {"fg_color": ("gray80", "gray20"), "hover_color": ("gray70", "gray30"), "corner_radius": 8}
    entry_style = {"fg_color": ("gray87", "gray13"), "corner_radius": 8}
    option_menu_style = {"fg_color": ("gray87", "gray13"), "button_color": ("gray80", "gray20"), "button_hover_color": ("gray70", "gray30"), "corner_radius": 8}
    
    def toggle_grid_entries():
        state = "disabled" if auto_mode_var.get() else "normal"
        rows_entry.configure(state=state)
        cols_entry.configure(state=state)

    def select_gif():
        filepath = filedialog.askopenfilename(filetypes=[("GIF files", "*.gif")])
        if filepath:
            try:
                total_frames, (width, height) = validate_gif(filepath)
                gif_path_var.set(filepath)
                status_label.configure(text=f"📁 Selected: {os.path.basename(filepath)} ({width}x{height}, {total_frames} frames)")
                show_preview(filepath)
                start_frame_entry.delete(0, "end")
                start_frame_entry.insert(0, "0")
                end_frame_entry.delete(0, "end")
                end_frame_entry.insert(0, str(total_frames))
            except ValueError as e:
                show_error("Invalid GIF", str(e))
                status_label.configure(text="❌ Invalid GIF file selected.")

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
            show_error("Preview Error", f"Failed to show preview: {str(e)}")
            status_label.configure(text=f"❌ Preview failed: {e}")

    def select_output_dir():
        path = filedialog.askdirectory()
        if path:
            output_dir_var.set(path)
            status_label.configure(text=f"Outputting to: {path}")

    def start_processing():
        gif_path, output_dir = gif_path_var.get(), output_dir_var.get()
        if not gif_path or not output_dir:
            show_error("Missing Information", "Please select both a GIF and an output folder.")
            return

        try:
            start_frame, end_frame = int(start_frame_entry.get()), int(end_frame_entry.get())
            if start_frame < 0 or end_frame <= start_frame: 
                raise ValueError("Frame range is invalid.")
        except ValueError:
            show_error("Invalid Frame Range", "Please enter valid frame numbers. End frame must be greater than start frame.")
            return

        rows, cols, auto_mode = 3, 3, auto_mode_var.get()
        if not auto_mode:
            try:
                rows, cols = int(rows_entry.get()), int(cols_entry.get())
                if rows <= 0 or cols <= 0: raise ValueError
            except ValueError:
                show_error("Invalid Grid", "Rows and columns must be positive integers.")
                return

        selected_tile = None
        if tile_text := single_tile_entry.get().strip():
            try:
                r, c = map(int, tile_text.split(","))
                selected_tile = (r, c)
            except (ValueError, IndexError):
                show_error("Invalid Tile", "Please use format: row,col (e.g., 0,1)")
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

        # Use thread manager instead of direct thread creation
        thread = thread_manager.start_thread(
            target=gif2grid_zip,
            args=(
                gif_path, os.path.join(output_dir, "gif_tiles_export.zip"),
                lambda val: app.after(0, update_gui, progress_bar, val)
            ),
            name="gif_processor"
        )

    # Initialize variables
    gif_path_var = ctk.StringVar()
    output_dir_var = ctk.StringVar()
    zip_only_var = ctk.BooleanVar(value=False)
    auto_mode_var = ctk.BooleanVar(value=False)
    
    # File selection buttons
    ctk.CTkButton(scrollable_splitter_frame, text="📂 Select GIF", command=select_gif, **button_style).pack(pady=5, padx=10, fill="x")
    ctk.CTkButton(scrollable_splitter_frame, text="📁 Select Output Folder", command=select_output_dir, **button_style).pack(pady=5, padx=10, fill="x")
    
    # Preview frame
    preview_frame = ctk.CTkFrame(scrollable_splitter_frame, fg_color=("gray82", "gray18"), corner_radius=10)
    preview_frame.pack(pady=10, padx=10, fill="x")
    preview_label = ctk.CTkLabel(preview_frame, text="GIF Preview will appear here")
    preview_label.pack(pady=10)

    # Frame range frame
    frame_range_frame = ctk.CTkFrame(scrollable_splitter_frame, fg_color=("gray82", "gray18"), corner_radius=10)
    frame_range_frame.pack(pady=5, padx=10, fill="x")
    frame_range_frame.columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(frame_range_frame, text="Start Frame:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
    start_frame_entry = ctk.CTkEntry(frame_range_frame, justify="center", **entry_style)
    start_frame_entry.grid(row=1, column=0, padx=5, pady=2, sticky="ew")
    ctk.CTkLabel(frame_range_frame, text="End Frame:").grid(row=0, column=1, padx=5, pady=2, sticky="w")
    end_frame_entry = ctk.CTkEntry(frame_range_frame, justify="center", **entry_style)
    end_frame_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

    # Filter menu
    filter_frame = ctk.CTkFrame(scrollable_splitter_frame, fg_color=("gray82", "gray18"), corner_radius=10)
    filter_frame.pack(pady=5, padx=10, fill="x")
    ctk.CTkLabel(filter_frame, text="Filter:").pack(pady=(10, 2))
    filter_menu = ctk.CTkOptionMenu(filter_frame, values=["None", "Grayscale", "Tint"], **option_menu_style)
    filter_menu.pack(pady=(0, 10))

    # Grid settings
    grid_settings_frame = ctk.CTkFrame(scrollable_splitter_frame, fg_color=("gray82", "gray18"), corner_radius=10)
    grid_settings_frame.pack(pady=5, padx=10, fill="x")
    ctk.CTkLabel(grid_settings_frame, text="Grid Settings:", font=("Arial", 14, "bold")).pack(pady=5)
    ctk.CTkCheckBox(grid_settings_frame, text="🧠 Auto Grid", variable=auto_mode_var, command=toggle_grid_entries).pack(pady=5)
    
    grid_frame = ctk.CTkFrame(grid_settings_frame, fg_color=("gray85", "gray15"), corner_radius=8)
    grid_frame.pack(pady=5, padx=10, fill="x")
    grid_frame.columnconfigure((0, 1), weight=1)
    ctk.CTkLabel(grid_frame, text="Rows:").grid(row=0, column=0)
    rows_entry = ctk.CTkEntry(grid_frame, width=80, justify="center", **entry_style)
    rows_entry.insert(0, "3")
    rows_entry.grid(row=1, column=0)
    ctk.CTkLabel(grid_frame, text="Columns:").grid(row=0, column=1)
    cols_entry = ctk.CTkEntry(grid_frame, width=80, justify="center", **entry_style)
    cols_entry.insert(0, "3")
    cols_entry.grid(row=1, column=1)

    # Single tile entry
    tile_frame = ctk.CTkFrame(scrollable_splitter_frame, fg_color=("gray82", "gray18"), corner_radius=10)
    tile_frame.pack(pady=5, padx=10, fill="x")
    ctk.CTkLabel(tile_frame, text="Optional: Only Export One Tile").pack(pady=(10, 0))
    single_tile_entry = ctk.CTkEntry(tile_frame, width=140, placeholder_text="row,col (e.g. 0,1)", **entry_style)
    single_tile_entry.pack(pady=(0, 10))
    toggle_grid_entries()

    # Start button
    start_button = ctk.CTkButton(scrollable_splitter_frame, text="▶️ Start Splitting", command=start_processing, height=40, **button_style)
    start_button.pack(pady=20, padx=10, fill="x")
    
    # Progress bar
    progress_frame = ctk.CTkFrame(scrollable_splitter_frame, fg_color=("gray82", "gray18"), corner_radius=10)
    progress_frame.pack(pady=5, padx=10, fill="x")
    progress_bar = ctk.CTkProgressBar(progress_frame)
    progress_bar.set(0)
    progress_bar.pack(pady=10, padx=10, fill="x")
    
    # Status label
    status_frame = ctk.CTkFrame(scrollable_splitter_frame, fg_color=("gray82", "gray18"), corner_radius=10)
    status_frame.pack(pady=5, padx=10, fill="x")
    status_label = ctk.CTkLabel(status_frame, text="Select a GIF and output folder to begin.", wraplength=500)
    status_label.pack(pady=10)

    # ===================================================================================
    # --- ROBLOX UPLOADER TAB
    # ===================================================================================
    
    scrollable_uploader_frame = ctk.CTkScrollableFrame(uploader_tab, fg_color=("gray87", "gray13"), corner_radius=10)
    scrollable_uploader_frame.pack(fill="both", expand=True, padx=10, pady=10)

    log_file_path_var = ctk.StringVar()

    def select_upload_files():
        global selected_upload_files
        files = list(filedialog.askopenfilenames(
            title="Select files to upload",
            filetypes=[("Assets", "*.png *.jpg *.mp3 *.ogg *.fbx"), ("All files", "*.*")]
        ))
        if files:
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
            show_error("Missing Dependency", 
                      "rblx-open-cloud library not found. Please install it using:\npip install rblx-open-cloud")
            return

        key, owner_type, owner_id_str = api_key_entry.get().strip(), owner_var.get(), id_entry.get().strip()
        
        if not key or not selected_upload_files or not owner_id_str:
            show_error("Missing Information", 
                      "Please provide:\n- API Key\n- Owner ID\n- Files to upload")
            return
        
        if not EnhancedFileValidator.validate_api_key(key):
            show_error("Invalid API Key", "The provided API key format is invalid.")
            return
        
        try:
            owner_id = int(owner_id_str)
        except ValueError:
            show_error("Invalid ID", "Owner ID must be a number")
            return

        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
        valid_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.tga', '.bmp', '.mp3', '.ogg', '.wav', '.flac', '.fbx'}
        
        # Validate all files before starting upload
        invalid_files = []
        oversized_files = []
        
        for file_path in selected_upload_files:
            if not os.path.exists(file_path):
                invalid_files.append(f"{os.path.basename(file_path)} (file not found)")
                continue
            
            if not EnhancedFileValidator.validate_file_type(file_path, valid_extensions):
                invalid_files.append(f"{os.path.basename(file_path)} (unsupported file type)")
                continue
            
            if not EnhancedFileValidator.validate_file_size(file_path, MAX_FILE_SIZE):
                oversized_files.append(os.path.basename(file_path))

        if invalid_files:
            show_error("Invalid Files", 
                      f"Invalid or unsupported files:\n{', '.join(invalid_files)}")
            return

        if oversized_files:
            show_error("File Size Error", 
                      f"Files exceed 10MB limit:\n{', '.join(oversized_files)}")
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
                    show_error("Authentication Error", "Failed to authenticate with Roblox", str(e))
                    start_upload_button.configure(state="normal")
                app.after(0, update_on_error)
                return

            total = len(selected_upload_files)
            successes = 0
            upload_results = []
            errors = []

            for i, path in enumerate(selected_upload_files):
                ext = os.path.splitext(path)[1].lower()
                atype = (AssetType.Decal if ext in ['.png', '.jpg', '.jpeg', '.gif', '.tga', '.bmp'] else
                         AssetType.Audio if ext in ['.mp3', '.ogg', '.wav', '.flac'] else
                         AssetType.Model if ext == '.fbx' else None)
                if not atype: 
                    print(f"Warning: Skipping unsupported file type: {path}")
                    continue

                def upload_with_retry():
                    with open(path, "rb") as file:
                        file_name = os.path.splitext(os.path.basename(path))[0]
                        return creator.upload_asset(file, atype, file_name, "Uploaded via Multi-Tool 2.").wait()

                try:
                    asset = EnhancedRetryManager.retry_operation(upload_with_retry)
                    successes += 1
                    upload_results.append(f"{os.path.basename(path)},{asset.id}")
                    print(f"Uploaded {os.path.basename(path)} as Asset ID: {asset.id}")
                except Exception as e:
                    errors.append(f"Failed to upload {os.path.basename(path)}: {str(e)}")
                
                app.after(0, upload_progress.set, (i + 1) / total)

            # Save log file if a path is provided
            log_save_status = ""
            if log_path := log_file_path_var.get():
                try:
                    # Create backup of existing log file
                    if os.path.exists(log_path):
                        backup_path = EnhancedBackupManager(max_backups=5).create_backup(log_path)
                        if backup_path:
                            log_save_status = f"\nCreated backup of existing log at: {os.path.basename(backup_path)}"

                    with open(log_path, "w") as log_file:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_file.write(f"# Roblox Asset Upload Log\n")
                        log_file.write(f"# Generated: {timestamp}\n")
                        log_file.write("Filename,AssetID\n")
                        for result_line in upload_results:
                            log_file.write(f"{result_line}\n")
                        if errors:
                            log_file.write("\n# Errors:\n")
                            for error in errors:
                                log_file.write(f"# {error}\n")
                    log_save_status += f"\nLog saved to {os.path.basename(log_path)}."
                except Exception as e:
                    log_save_status = f"\n❌ Error saving log: {e}"
                    print(f"Error writing to log file '{log_path}': {e}")
                    
                    # Try to restore from backup if available
                    if backup_path and os.path.exists(backup_path):
                        if EnhancedBackupManager(max_backups=5).restore_backup(backup_path, log_path):
                            log_save_status += "\nRestored from backup."
            
            def update_on_finish():
                final_msg = f"✅ Done: {successes}/{total} uploaded."
                if errors:
                    final_msg += f"\n❌ {len(errors)} errors occurred."
                    show_error("Upload Complete", 
                              f"Upload completed with {len(errors)} errors.\n\n" + 
                              "\n".join(errors[:5]) + 
                              ("\n...and more" if len(errors) > 5 else ""))
                final_msg += log_save_status
                upload_status.configure(text=final_msg)
                start_upload_button.configure(state="normal")

            app.after(0, update_on_finish)

        # Use thread manager instead of direct thread creation
        thread = thread_manager.start_thread(
            target=task,
            name="uploader"
        )
    
    # API Key section
    api_frame = ctk.CTkFrame(scrollable_uploader_frame, fg_color=("gray82", "gray18"), corner_radius=10)
    api_frame.pack(fill="x", padx=10, pady=5)
    ctk.CTkLabel(api_frame, text="Roblox API Key:", font=("Arial", 14)).pack(pady=(10,2), padx=10, anchor="w")
    api_key_entry = ctk.CTkEntry(api_frame, placeholder_text="Enter your Open Cloud API Key", show="*", **entry_style)
    api_key_entry.pack(fill="x", padx=10, pady=(0, 10))

    # Owner section
    owner_frame = ctk.CTkFrame(scrollable_uploader_frame, fg_color=("gray82", "gray18"), corner_radius=10)
    owner_frame.pack(fill="x", padx=10, pady=5)
    ctk.CTkLabel(owner_frame, text="Owner (Upload Destination):", font=("Arial", 14)).pack(pady=(10,2), padx=10, anchor="w")
    owner_var = ctk.StringVar(value="Group")
    owner_type_menu = ctk.CTkSegmentedButton(owner_frame, values=["User", "Group"], variable=owner_var)
    owner_type_menu.pack(side="left", padx=(10, 10), pady=(0, 10))
    id_entry = ctk.CTkEntry(owner_frame, placeholder_text="Enter User ID or Group ID", **entry_style)
    id_entry.pack(fill="x", padx=10, pady=(0, 10))

    # File selection
    ctk.CTkButton(scrollable_uploader_frame, text="📂 Select Files to Upload", command=select_upload_files, **button_style).pack(pady=20, padx=10, fill="x")

    # Log file section
    log_frame = ctk.CTkFrame(scrollable_uploader_frame, fg_color=("gray82", "gray18"), corner_radius=10)
    log_frame.pack(fill="x", padx=10, pady=5)
    ctk.CTkLabel(log_frame, text="Optional: Save Asset IDs to Log File", font=("Arial", 14)).pack(pady=(10,2), padx=10, anchor="w")
    log_file_button = ctk.CTkButton(log_frame, text="Set Log File", command=select_log_file, width=120, **button_style)
    log_file_button.pack(side="left", padx=(10, 10), pady=(0, 10))
    log_file_entry = ctk.CTkEntry(log_frame, textvariable=log_file_path_var, placeholder_text="No log file selected", state="disabled", **entry_style)
    log_file_entry.pack(side="left", fill="x", expand=True, padx=(0, 10), pady=(0, 10))

    # Upload button
    start_upload_button = ctk.CTkButton(scrollable_uploader_frame, text="🚀 Start Upload", command=start_upload, height=40, **button_style)
    start_upload_button.pack(pady=10, padx=10, fill="x")

    # Progress bar
    upload_progress_frame = ctk.CTkFrame(scrollable_uploader_frame, fg_color=("gray82", "gray18"), corner_radius=10)
    upload_progress_frame.pack(pady=5, padx=10, fill="x")
    upload_progress = ctk.CTkProgressBar(upload_progress_frame)
    upload_progress.set(0)
    upload_progress.pack(pady=10, padx=10, fill="x")

    # Status label
    upload_status_frame = ctk.CTkFrame(scrollable_uploader_frame, fg_color=("gray82", "gray18"), corner_radius=10)
    upload_status_frame.pack(pady=5, padx=10, fill="x")
    upload_status = ctk.CTkLabel(upload_status_frame, text="Enter API info and select files.", wraplength=500)
    upload_status.pack(pady=10)

    # Add cleanup on window close
    def on_closing():
        thread_manager.cleanup()
        app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_closing)

    app.mainloop()

if __name__ == "__main__":
    run_gui()

class PerformanceMetrics:
    """Tracks and reports performance metrics."""
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}
    
    def record_metric(self, name: str, value: float):
        """Record a performance metric."""
        self.metrics[name] = value
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all recorded metrics."""
        return self.metrics.copy()

class CacheManager:
    """Manages file and image caching."""
    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self.max_size = max_size
        self.cache = {}
        self.access_count = {}
    
    def get(self, key: str) -> Any:
        """Get an item from cache."""
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """Add an item to cache."""
        if len(self.cache) >= self.max_size:
            self._evict_least_used()
        self.cache[key] = value
        self.access_count[key] = 1
    
    def _evict_least_used(self):
        """Remove least accessed item from cache."""
        if not self.cache:
            return
        min_key = min(self.access_count.items(), key=lambda x: x[1])[0]
        del self.cache[min_key]
        del self.access_count[min_key]

class FileChangeHandler(FileSystemEventHandler):
    """Handles file system events."""
    def __init__(self, callback: Callable):
        self.callback = callback
    
    def on_modified(self, event):
        if not event.is_directory:
            self.callback(event.src_path)

class FileWatcher:
    """Watches for file changes and triggers callbacks."""
    def __init__(self):
        self.observer = Observer()
        self.watched_paths = set()
    
    def watch(self, path: str, callback: Callable):
        """Watch a path for changes."""
        if path not in self.watched_paths:
            self.observer.schedule(FileChangeHandler(callback), path, recursive=False)
            self.watched_paths.add(path)
            if not self.observer.is_alive():
                self.observer.start()
    
    def stop(self):
        """Stop watching all paths."""
        self.observer.stop()
        self.observer.join()
        self.watched_paths.clear()

class ImageEnhancer:
    """Provides advanced image processing capabilities."""
    def __init__(self):
        self.cache = CacheManager()
    
    def enhance(self, image: Image.Image, enhancement_type: str) -> Image.Image:
        """Apply image enhancement."""
        cache_key = f"{hash(str(image.tobytes()))}_{enhancement_type}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        if enhancement_type == "sharpen":
            result = image.filter(ImageFilter.SHARPEN)
        elif enhancement_type == "blur":
            result = image.filter(ImageFilter.BLUR)
        elif enhancement_type == "contrast":
            enhancer = ImageEnhance.Contrast(image)
            result = enhancer.enhance(1.5)
        else:
            result = image
        
        self.cache.put(cache_key, result)
        return result

class AsyncFileHandler:
    """Handles file operations asynchronously."""
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    async def read_file(self, path: str) -> bytes:
        """Read file asynchronously."""
        async with aiofiles.open(path, 'rb') as f:
            return await f.read()
    
    async def write_file(self, path: str, data: bytes):
        """Write file asynchronously."""
        async with aiofiles.open(path, 'wb') as f:
            await f.write(data)

class SignalHandler:
    """Manages system signals for graceful shutdown."""
    def __init__(self):
        self.original_handlers = {}
        self.cleanup_handlers = []
    
    def register_cleanup(self, handler: Callable):
        """Register a cleanup handler."""
        self.cleanup_handlers.append(handler)
    
    def setup(self):
        """Setup signal handlers."""
        signals = [signal.SIGINT, signal.SIGTERM]
        for sig in signals:
            self.original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """Handle system signals."""
        for handler in self.cleanup_handlers:
            try:
                handler()
            except Exception as e:
                logging.error(f"Cleanup handler failed: {e}")
        sys.exit(0)

class ProgressTracker:
    """Tracks and displays progress with enhanced features."""
    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.current = 0
        self.desc = desc
        self.start_time = time.time()
        self.pbar = tqdm(total=total, desc=desc)
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        self.pbar.update(increment)
    
    def close(self):
        """Close progress bar."""
        self.pbar.close()

class ResourceManager:
    """Manages system resources and cleanup."""
    def __init__(self):
        self.resources = []
        self.cleanup_handlers = []
    
    def register_resource(self, resource: Any, cleanup: Callable):
        """Register a resource with cleanup handler."""
        self.resources.append(resource)
        self.cleanup_handlers.append(cleanup)
    
    def cleanup(self):
        """Clean up all registered resources."""
        for handler in reversed(self.cleanup_handlers):
            try:
                handler()
            except Exception as e:
                logging.error(f"Resource cleanup failed: {e}")

class ConfigManager:
    """Manages application configuration."""
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_config(self):
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
        self.save_config()

class ErrorHandler:
    """Provides enhanced error handling with recovery options."""
    def __init__(self):
        self.error_handlers = {}
        self.recovery_strategies = {}
    
    def register_handler(self, error_type: type, handler: Callable):
        """Register an error handler."""
        self.error_handlers[error_type] = handler
    
    def register_recovery(self, error_type: type, strategy: Callable):
        """Register a recovery strategy."""
        self.recovery_strategies[error_type] = strategy
    
    def handle_error(self, error: Exception) -> bool:
        """Handle an error using registered handlers and recovery strategies."""
        error_type = type(error)
        
        # Try error handler
        if error_type in self.error_handlers:
            self.error_handlers[error_type](error)
        
        # Try recovery strategy
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](error)
        
        return False

# Create global instances
performance_metrics = PerformanceMetrics()
cache_manager = CacheManager()
file_watcher = FileWatcher()
image_enhancer = ImageEnhancer()
async_file_handler = AsyncFileHandler()
signal_handler = SignalHandler()
resource_manager = ResourceManager()
config_manager = ConfigManager()
error_handler = ErrorHandler()
