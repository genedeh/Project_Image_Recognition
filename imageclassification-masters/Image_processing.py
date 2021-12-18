import numpy as np
import time
import os
from os import path
from PIL import Image, ImageEnhance
import imageio
import concurrent.futures
from Folder_processing import FolderFileProcessing
from Face_detection import FaceDetection
from Image_clasification import ImageClassification


class ImageProcessing(FolderFileProcessing, FaceDetection, ImageClassification):
    get_current_working_directory = os.getcwd()
    imagesPath = f'{get_current_working_directory}/437/'
    reprocessPath = f'{get_current_working_directory}/Reports/reprocess_images/'
    imageReport = f'{get_current_working_directory}/Reports/'
    primaryFolders = ['good_images', 'face_in_background', 'no_human_face', 'defaced', 'not_sure', 'reprocess_images']
    re_run = 0
    dark_images = 0

    def __init__(self):
        if not path.exists(self.imageReport):
            os.mkdir(self.imageReport)
        else:
            self.delete_directories(self.imageReport)
        self.create_directories(self.primaryFolders)

    def check_light_dark(self, img):
        f = imageio.imread(img, as_gray=True)
        is_light = np.mean(f) > 127
        return 'light' if is_light else 'dark'

    def enhance_image(self, img, num):
        im = Image.open(img)
        enhancer = ImageEnhance.Brightness(im)
        enhanced_im = enhancer.enhance(num)
        enhanced_im.save("./sample.jpg")

    def initiate_threading(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            executor.submit(self.verifyReprocessingImages())

    def verify_reprocessing_images(self, good=None, not_sure=None, another_face=None):
        start_time = time.time()
        print('----- INITIATE VERIFICATION OF NEED TO BE REPROCESSED IMAGES  ------\n')
        for root, directories, files in os.walk(self.reprocessPath, topdown=False):
            for image in files:
                if ('.jpg' in image) or ('.jpeg' in image):
                    temp_result = self.detect_face_fensor_flow(f'{root}/{image}')
                    if len(temp_result) > 0:
                        if temp_result[0] > 50:
                            good += 1
                            self.create_good_images(root, f'{root}/{image}')
                        elif len(temp_result) > 1:
                            another_face += 1
                            self.create_face_background(root, f'{root}/{image}')
                        else:
                            not_sure += 1
                            self.create_not_sure_images(root, f'{root}/{image}')

                    elif len(temp_result) == 0:
                        case = ''
                        percent = 0
                        for c in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]:
                            self.enhance_image(root + '/' + image, c)
                            _result = self.detect_face_tensor_flow("./sample.jpg")
                            print(_result, 'first=>   ', c)
                            if len(_result) > 0:
                                if _result[0] > 50:
                                    case = 'good'
                                    percent = _result[0]
                                    break
                                elif 10 < _result[0] < 50:
                                    case = 'not_sure'
                                    percent = _result[0]
                                    break
                                else:
                                    __result = self.detect_face_model("./sample.jpg")
                                    print(__result, 'second=>   ', c)
                                    if len(__result) > 0:
                                        if __result[0] > 50:
                                            case = 'good'
                                            percent = __result[0]
                                            break
                                        elif 10 < __result[0] < 50:
                                            case = 'not_sure'
                                            percent = __result[0]
                                            break
                                    else:
                                        case = 'no_human_face'
                                        percent = 0
                                        break
                            else:
                                case = 'no_human_face'
                                percent = 0
                                break

                        print(percent)
                        print(case)

        print('------ ', time.time() - start_time, ' -----\n')


image_processing = ImageProcessing()
image_processing.classify_images()
