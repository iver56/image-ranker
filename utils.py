import os


def get_image_file_paths(image_root_path="data"):
    """Return a list of paths to all image files in a directory.

    Does not go into subdirectories.
    """
    image_file_paths = []

    for root, dirs, filenames in os.walk(image_root_path):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)

            file_extension = filename.split(".")[-1]
            if file_extension.lower() in ("png", "jpg", "jpeg"):
                image_file_paths.append(file_path)

        break  # prevent descending into subfolders

    return image_file_paths
