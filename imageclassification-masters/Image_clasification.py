import os
import time

start_time = time.time()


class ImageClassification:
    def classify_images(self):
        print('Classifying Faces In Images...')
        good = 0
        another_face = 0
        no_human_face = 0
        not_sure = 0
        reprocess_images = 0 
        for root, directories, files in os.walk(self.imagesPath, topdown=False):
            for image in files:  # or ('.jpg' in image) or ('.jpeg' in image)
                if ('.jpg' in image) or ('.jpeg' in image):
                    light_dark = self.check_light_dark(f'{root}/{image}')
                    if light_dark == 'dark':
                        self.dark_images += 1
                        self.enhance_image(f'{root}/{image}', 2.5)
                        result = self.detect_face_model("./sample.jpg")
                    else:
                        result = self.detect_face_model(f'{root}/{image}')

                    # GOOD IMAGES CLASSIFIED AS 95 FOR FIRST CHECK
                    if len(result) == 1 and result[0] > 75:
                        good += 1
                        self.create_good_images(root, f'{root}/{image}')

                    # IMAGES WITH ANOTHER FACE CHECK CLASSIFIED BELOW 75 THEN RUN OTHER CHECKS
                    elif len(result) > 1 and result[0] < 75:
                        temp_result = self.detect_face_tensor_flow(f'{root}/{image}')
                        if len(temp_result) > 0:
                            if len(temp_result) == 1 and result[0] > temp_result[0]:
                                good += 1
                                self.create_good_images(root, f'{root}/{image}')
                            elif temp_result[0] > result[0] and len(temp_result) == 1:
                                good += 1
                                self.create_good_images(root, f'{root}/{image}')
                            else:
                                no_human_face += 1
                                self.create_no_human_face(root, f'{root}/{image}')

                    # IMAGES WITH ANOTHER FACE CHECK CLASSIFIED ABOVE 95 THEN RUN OTHER CHECKS
                    elif len(result) > 1 and result[1] > 85:
                        temp_result = self.detect_face_tensor_flow(f'{root}/{image}')
                        if len(temp_result) > 0:
                            if temp_result[0] > 75:
                                good += 1
                                self.create_good_images(root, f'{root}/{image}')
                            elif len(temp_result) > 1:
                                another_face += 1
                                self.create_face_background(root, f'{root}/{image}')
                            elif 10 < temp_result[0] < 50:
                                not_sure += 1
                                self.create_not_sure_images(root, f'{root}/{image}')
                            else:
                                no_human_face += 1
                                self.create_no_human_face(root, f'{root}/{image}')

                        elif result[0] > 75 and len(temp_result) == 0:
                            good += 1
                            self.create_good_images(root, f'{root}/{image}')
                        else:
                            another_face += 1
                            self.create_face_background(root, f'{root}/{image}')

                    # IMAGES WITH FIRST Detection AS TWO FACE BUT NEXT Detection AS ONE FACE
                    elif len(result) > 1 and result[0] > 75:
                        temp_result = self.detect_face_tensor_flow(f'{root}/{image}')
                        if len(result) > 1 and len(temp_result) > 1:
                            another_face += 1
                            self.create_face_background(root, f'{root}/{image}')
                        elif len(temp_result) > 0:
                            if temp_result[0] > 75:
                                good += 1
                                self.create_good_images(root, f'{root}/{image}')
                            elif result[0] > temp_result[0]:
                                good += 1
                                self.create_good_images(root, f'{root}/{image}')
                            elif temp_result[0] > result[0]:
                                good += 1
                                self.create_good_images(root, f'{root}/{image}')
                            else:
                                not_sure += 1
                                self.create_not_sure_images(root, f'{root}/{image}')
                        elif result[0] > 50:
                            good += 1
                            self.create_good_images(root, f'{root}/{image}')
                        elif 10 < result[0] < 50:
                            not_sure += 1
                            self.create_not_sure_images(root, f'{root}/{image}')
                        else:
                            no_human_face += 1
                            self.create_no_human_face(root, f'{root}/{image}')


                    # NO FACE AT ALL
                    elif len(result) == 0:
                        reprocess_images += 1
                        self.re_process_human_face(root, f'{root}/{image}')
                        pass

                    else:
                        temp_result = self.detect_face_tensor_flow(f'{root}/{image}')
                        if len(temp_result) > 0:
                            if temp_result[0] > 75:
                                good += 1
                                self.create_good_images(root, f'{root}/{image}')
                            elif temp_result[0] > result[0]:
                                good += 1
                                self.create_good_images(root, f'{root}/{image}')
                            elif result[0] > temp_result[0]:
                                good += 1
                                self.create_good_images(root, f'{root}/{image}')

                        elif result[0] > 50:
                            good += 1
                            self.create_good_images(root, f'{root}/{image}')
                        elif 10 < result[0] < 50:
                            not_sure += 1
                            self.create_not_sure_images(root, f'{root}/{image}')
                        else:
                            no_human_face += 1
                            self.create_no_human_face(root, f'{root}/{image}')

        print('Good: ', good, '| Another Face: ', another_face, '| Not Sure: ', not_sure, '| No Human Face: ',
              no_human_face, '| Reverification Needed:  ', reprocess_images, '| Reprocessed: ', self.re_run,
              '| Dark Images: ', self.dark_images, '\n')
        print(f'------ Process Took {time.time() - start_time} -----\n')
