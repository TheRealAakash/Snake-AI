import glob
import os
import shutil

dirs = ['past_games', 'logs', 'models']
exclude = []
exclude_proj = []
exclude_dirs = []
dirs = list(set(dirs).difference(exclude_dirs))
delete_all = input(f"Do you want to delete all files: ").lower()
delete_all = delete_all == "y" or delete_all == "yes"
bkup = False
if not delete_all:
    bkup = input(f"Do you want to bkup all files: ").lower()
    bkup = bkup == "y" or bkup == "yes"
if bkup:
    name = input("Where do you want to save it: ")
else:
    name = None
for directory in dirs:
    files = [f for f in glob.glob(f"{directory}/*", recursive=True)]
    to_exclude = [f"{directory}\\{path}" for path in exclude]

    files = list(set(files).difference(to_exclude))
    exclude_files = []
    for file in files:
        for project in exclude_proj:
            if project in file:
                exclude_files.append(file)

    files = list(set(files).difference(exclude_files))
    delete = True  # input(f"Do you want to delete all files in {directory}: ").lower() == 'yes'
    if bkup:
        shutil.move(f"{directory}", f"bkup/{name}")
    if delete_all:
        for file in files:
            if os.path.isfile(file):
                os.remove(file)
            else:
                shutil.rmtree(file)
os.system("cls")
