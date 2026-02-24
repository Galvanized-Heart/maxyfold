import os

# CONFIGURATION
# Only look at these top level paths
TARGETS = ['src', 'configs', 'scripts', 'data', 'tests', 'pyproject.toml', 'Makefile']

# Only read files with these extensions
Keep_Exts = {'.py', '.toml', '.yaml', '.yml', '.md', '.json', '.sh', '.txt'}

# Skip files larger than 100KB
MAX_FILE_SIZE = 100 * 1024 
OUTPUT_FILE = 'context.txt'

def is_text_file(filename):
    return any(filename.endswith(ext) for ext in Keep_Exts)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
    # 1. WRITE STRUCTURE
    out.write("DIRECTORY STRUCTURE:\n====================\n")
    for target in TARGETS:
        if not os.path.exists(target): continue
        if os.path.isfile(target):
            out.write(f"{target}\n")
        else:
            # Simple tree walker for specific folders
            for root, dirs, files in os.walk(target):
                # Skip hidden folders like .ipynb_checkpoints inside your targets
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                level = root.count(os.sep)
                indent = ' ' * 4 * level
                out.write(f"{indent}{os.path.basename(root)}/\n")
                subindent = ' ' * 4 * (level + 1)
                for f in files:
                    out.write(f"{subindent}{f}\n")

    # 2. WRITE CONTENT
    out.write("\n\nFILE CONTENTS:\n====================\n")
    for target in TARGETS:
        if not os.path.exists(target): continue
        
        # Handle if target is a single file (pyproject.toml)
        if os.path.isfile(target):
            out.write(f"\n--- START: {target} ---\n")
            with open(target, 'r', encoding='utf-8') as f:
                out.write(f.read())
            out.write(f"\n--- END: {target} ---\n")
            continue

        # Handle directories
        for root, dirs, files in os.walk(target):
            dirs[:] = [d for d in dirs if not d.startswith('.')] # Skip hidden
            
            for f in files:
                if not is_text_file(f): continue
                
                path = os.path.join(root, f)
                try:
                    if os.path.getsize(path) > MAX_FILE_SIZE:
                        out.write(f"\n[SKIPPED LARGE FILE: {path}]\n")
                        continue
                        
                    with open(path, 'r', encoding='utf-8') as infile:
                        out.write(f"\n{'='*30}\nFILE: {path}\n{'='*30}\n")
                        out.write(infile.read())
                except Exception as e:
                    out.write(f"\n[ERROR READING {path}]\n")

print(f"Done! Download or copy content from: {OUTPUT_FILE}")