"""
Load all files from data/raw directory structure.
Supports CSV, MAT, and JPG files from subdirectories.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
import hashlib


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash for a file."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_file_info(file_path: Path) -> Dict:
    """Extract basic file information."""
    return {
        'name': file_path.name,
        'path': str(file_path.absolute()),
        'size': file_path.stat().st_size,
        'extension': file_path.suffix.lower(),
        'hash': compute_file_hash(str(file_path))
    }


def load_files_from_directory(directory: str, extensions: Optional[List[str]] = None) -> List[Dict]:
    """
    Load all files from a directory with optional extension filtering.
    
    Args:
        directory: Path to directory to scan
        extensions: List of file extensions to include (e.g., ['.csv', '.mat'])
    
    Returns:
        List of file information dictionaries
    """
    if not os.path.exists(directory):
        print(f"⚠️  Directory not found: {directory}")
        return []
    
    files = []
    dir_path = Path(directory)
    
    # Get all files in directory and subdirectories
    for file_path in dir_path.rglob('*'):
        if file_path.is_file():
            # Filter by extensions if specified
            if extensions and file_path.suffix.lower() not in extensions:
                continue
            
            try:
                file_info = get_file_info(file_path)
                files.append(file_info)
                print(f"📁 Found: {file_info['name']} ({file_info['size']} bytes)")
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
    
    return files


def load_all_raw_files(base_dir: str = "data/raw") -> Dict[str, List[Dict]]:
    """
    Load all files from the raw data directory structure.
    
    Expected structure:
    data/raw/
    ├── csv/
    ├── mat/
    └── jpg/
    
    Returns:
        Dictionary with file types as keys and file lists as values
    """
    results = {
        'csv': [],
        'mat': [],
        'jpg': [],
        'other': []
    }
    
    # Define file type mappings
    type_mappings = {
        'csv': ['.csv'],
        'mat': ['.mat'],
        'jpg': ['.jpg', '.jpeg'],
    }
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"❌ Base directory not found: {base_dir}")
        return results
    
    print(f"🔍 Scanning {base_dir} for files...")
    
    # Load files by type from subdirectories
    for file_type, extensions in type_mappings.items():
        subdir = os.path.join(base_dir, file_type)
        files = load_files_from_directory(subdir, extensions)
        results[file_type] = files
        print(f"✅ Loaded {len(files)} {file_type.upper()} files")
    
    # Also scan for any other files in the base directory
    other_files = []
    base_path = Path(base_dir)
    all_known_extensions = [ext for exts in type_mappings.values() for ext in exts]
    
    for file_path in base_path.rglob('*'):
        if file_path.is_file():
            # Skip if it's in a subdirectory we already processed
            relative_path = file_path.relative_to(base_path)
            if len(relative_path.parts) > 1 and relative_path.parts[0] in type_mappings:
                continue
            
            # Skip if it's a known extension
            if file_path.suffix.lower() in all_known_extensions:
                continue
                
            try:
                file_info = get_file_info(file_path)
                other_files.append(file_info)
                print(f"📄 Other file: {file_info['name']}")
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
    
    results['other'] = other_files
    if other_files:
        print(f"ℹ️  Found {len(other_files)} other files")
    
    return results


def print_summary(files_dict: Dict[str, List[Dict]]) -> None:
    """Print a summary of loaded files."""
    print("\n" + "="*50)
    print("📊 FILE LOADING SUMMARY")
    print("="*50)
    
    total_files = 0
    total_size = 0
    
    for file_type, files in files_dict.items():
        if not files:
            continue
            
        type_size = sum(f['size'] for f in files)
        total_files += len(files)
        total_size += type_size
        
        print(f"{file_type.upper():>8}: {len(files):>4} files ({type_size:>10,} bytes)")
    
    print("-" * 50)
    print(f"{'TOTAL':>8}: {total_files:>4} files ({total_size:>10,} bytes)")
    print("="*50)


def main():
    """Main function to demonstrate usage."""
    # Load all files
    files = load_all_raw_files()
    
    # Print summary
    print_summary(files)
    
    # Example: Access specific file types
    csv_files = files['csv']
    mat_files = files['mat']
    jpg_files = files['jpg']
    
    # Example: Find files by hash (duplicate detection)
    print("\n🔍 Checking for duplicates...")
    all_files = [f for file_list in files.values() for f in file_list]
    hashes = [f['hash'] for f in all_files]
    
    if len(hashes) != len(set(hashes)):
        print("⚠️  Duplicate files detected!")
        hash_counts = {}
        for f in all_files:
            h = f['hash']
            if h in hash_counts:
                hash_counts[h].append(f)
            else:
                hash_counts[h] = [f]
        
        for h, file_list in hash_counts.items():
            if len(file_list) > 1:
                print(f"  Duplicate hash {h[:8]}...:")
                for f in file_list:
                    print(f"    - {f['path']}")
    else:
        print("✅ No duplicates found")
    
    return files


if __name__ == "__main__":
    main()