import os
import shutil

#this function moves all of the images inside train or test to within the main folder, 
# but renames the image to be the same format so that we can look it up in the csv
def move_and_rename_images(root_folder, destination_folder):
    for foldername, subfolders, filenames in os.walk(root_folder):
        for folder in subfolders:
            new_root_folder = f'{root_folder}/{folder}'
            for new_foldername, new_subfolders, new_filenames in os.walk(new_root_folder):
                for new_folder in new_subfolders:
                    new_new_root_folder = f'{new_root_folder}/{new_folder}'
                    for next_foldername, next_subfolders, next_filenames in os.walk(new_new_root_folder):
                        #print(f'Next Foldername: {next_foldername}')
                        #print(f'Next subfolders: {next_subfolders}')
                        #print(f'Next filenames : {next_filenames}\n')

           
                        for filename in next_filenames:
                            original_path = os.path.join(next_foldername, filename)
                            if filename.startswith('.'):
                                pass

                            try:
                                # Extract information from the original path
                                _, _, _, _, _, cell_line, plate = next_foldername.split(os.path.sep)

                                # Extract well information from the filename
                                well_info, extension = os.path.splitext(filename)
                                well, channel, rest = well_info.split('_')

                                # Remove 'Plate' from the plate information
                                plate_number = plate.replace('Plate', '')

                                # Construct the new filename
                                new_filename = f"{cell_line}_{plate_number}_{well}_{rest}{extension}"

                                # Construct the new destination path
                                new_destination_path = os.path.join(destination_folder, new_filename)

                                # Move and rename the file
                                shutil.move(original_path, new_destination_path)
                            except Exception as e:
                                print(f'{filename} not moved because of {e}')

# Example usage
# root_folder = 'data/train' #or test
# destination_folder = 'data/train' #or test
# move_and_rename_images(root_folder, destination_folder)

if __name__ == "__main__":
    print("Starting move_and_rename_image")
    print(f'Current directory: {os.getcwd()}')
    root_folder = '/home/dskinne3/CellularImage/train'
    destination_folder = '/home/dskinne3/CellularImage/train_fixed'
    move_and_rename_images(root_folder, destination_folder)
    print('Finished move and rename.')
