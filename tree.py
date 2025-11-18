import os

def print_tree(root, prefix=""):
    entries = sorted(os.listdir(root))
    for i, name in enumerate(entries):
        path = os.path.join(root, name)
        connector = "└── " if i == len(entries)-1 else "├── "
        print(prefix + connector + name)
        if os.path.isdir(path):
            extension = "    " if i == len(entries)-1 else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    print_tree(".")
