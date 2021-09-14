from abc import ABC
import shutil
import os
from os import path


class Directories(ABC):
    get_current_working_directory = os.getcwd()
    image_paths = f'{get_current_working_directory}/pictures/'
    image_report = f'{get_current_working_directory}/Reports/'

    def check_directory(self, _path):
        return path.exists(_path)

    def create_directory(self, folder):
        fol = self.strip_root_folder(folder)
        if path.exists(fol):
            pass
        else:
            try:
                os.makedirs(fol, 0o755, False)
            except OSError:
                pass

    def create_directories(self, arr):
        for folder in arr:
            try:
                os.mkdir(self.image_report + folder)
            except OSError:
                print("Creation of the directory %s failed" % folder)
            else:
                print("Successfully created the directory %s " % folder)

    def delete_directories(self, path):
        print(f'Attempting to delete...  {path}')
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f'Deleted ! {path}')
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def strip_root_folder(self, _path):
        assert _path is not None
        try:
            if self.image_paths in _path:
                return _path.replace(self.image_paths, '')
            else:
                return _path
        except:
            print(f'{self.image_paths}<==>{_path}')


class FolderFileProcessing(Directories):

    def create_good_images(self, root, path):
        if not self.check_directory(f'{self.image_report}good_images/{self.strip_root_folder(path)}'):
            self.create_directory(f'{self.image_report}good_images/{root}')
        shutil.copy(path, f'{self.image_report}good_images/{self.strip_root_folder(path)}')

    def create_not_sure_images(self, root, path):
        if not self.check_directory(f'{self.image_report}not_sure/{self.strip_root_folder(path)}'):
            self.create_directory(f'{self.image_report}not_sure/{root}')
        shutil.copy(path, f'{self.image_report}not_sure/{self.strip_root_folder(path)}')

    def create_face_background(self, root, path):
        if not self.check_directory(f'{self.image_report}face_in_background/{self.strip_root_folder(path)}'):
            self.create_directory(f'{self.image_report}face_in_background/{root}')
        shutil.copy(path, f'{self.image_report}face_in_background/{self.strip_root_folder(path)}')

    def create_no_human_face(self, root, path):
        if not self.check_directory(f'{self.image_report}no_human_face/{self.strip_root_folder(path)}'):
            self.create_directory(f'{self.image_report}no_human_face/{root}')
        shutil.copy(path, f'{self.image_report}no_human_face/{self.strip_root_folder(path)}')

    def re_process_human_face(self, root, path):
        if not self.check_directory(f'{self.image_report}reprocess_images/{self.strip_root_folder(path)}'):
            self.create_directory(f'{self.image_report}reprocess_images/{root}')
        shutil.copy(path, f'{self.image_report}reprocess_images/{self.strip_root_folder(path)}')
