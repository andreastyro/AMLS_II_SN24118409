# download_to_root_of_project.py
import os
import gdown

root_dir = os.path.dirname(os.path.abspath(__file__))

gdown.download_folder(
    url="https://drive.google.com/drive/folders/1lVD8AqBmaqSQTriCZqBjryKx2rwARrmI",
    output=root_dir,          # "."  = the directory you launch the script from
    quiet=False,         # progress bars
    use_cookies=False,   # works because the link is public
    remaining_ok=True    # keep fetching past the 50-item mark
)