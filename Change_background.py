#!/usr/bin/env python3
# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os, cv2, logging
import numpy as np
from os.path import exists
from ivit_i.common.app import iAPP_SEG

class Change_background(iAPP_SEG):
    def __init__(self,params:dict=None, label_path:str=None,):
        """
        Args:
            params (dict, optional): _description_. Defaults to None.
            label (str, optional): _description_. Defaults to None.
        """
        self.params = params
        self.label = self._check_label(label_path)
        self.background_frame_path = self._check_background_frame_path()
        self.change_object = self._check_change_object()

    def _check_label(self,label_path:str):
        """
        load all the label  according label path.
        Args:
            label_path (str): path to label.

        Raises:
            Error.FileNotFound: if path to label is not exist will crash.

        """
        
        if not exists(label_path):
            logging.error('Label path is incorrect: {}'.format(label_path))

        if isinstance(label_path, (list, tuple)):
            return label_path
        else:
            with open(label_path, 'r') as f:
                labels_map = [x.strip() for x in f]
            labels_map.append('background')
            return labels_map
            
    def _check_background_frame_path(self):
        """
            step1 : Verify app config.
            step2 : Verify background frame path.
            If all right , return path of background img.
        Returns:
            str : Return path of background img.
        """
        if not self.params.__contains__('application'):
            raise KeyError("App config isn't compliance with specifications!Don't have key 'application' in app config, please correct it!")
        
        if not self.params['application'].__contains__('background_frame'):
            raise KeyError("App config don't set background frame path! please correct it!")

        if not isinstance(self.params['application']['background_frame'],str):
            raise ValueError("Background frame path type must be str but you set {}! please correct it!".format(type(self.params['application']['background_frame'])))

        if not exists(self.params['application']['background_frame']):
            raise FileExistsError('image path is incorrect: {}'.format(self.params['application']['background_frame']))
        
        return self.params['application']['background_frame']

    def _check_change_object(self):
        """
        step1 : Verify app config.
        step2 : Verify change_object.
        If all right , return a list that contain objcet user want to change background.

        Returns:
            list: Return a list that contain objcet user want to change background. 
        """
        if not self.params['application'].__contains__('change_object'):
            raise KeyError("App config not set change object! please correct it!")
        
        if not isinstance(self.params['application']['change_object'],list):
            raise ValueError("Change object type must be list but you set {}! please correct it!".format(type(self.params['application']['background_frame'])))
        
        if len(self.params['application']['change_object'])!=0:
            return self.params['application']['change_object']
        else:
            return ['background']

    def create_mask(self,masks:np.ndarray):
        """
        step1 : Check change_object whether in label or not.
        step2 : Change_object fill 0. other fill 255.
 
        Args:
            masks (np.ndarray): output of model predict , shape must same with input image.

        Returns:
            masks (np.ndarray): background_mask
            ~masks (np.ndarray) : object_mask.
        """
       
        
        for label in self.change_object:
            if not (label in self.label):
                raise ValueError(" {} not in label!".format(label))
            masks[masks==(self.label.index(label))] = len(self.label)
        masks[masks<len(self.label)] = 255
        masks[masks==len(self.label)] = 0
        
        
        return masks,~masks

    def change_background_img(self,img_path:str):
        """
        If you want to change the background image you can use this function.

        Args:
            img_path (str): New background image path.

        """
        if isinstance(img_path,str):
            raise ValueError("The img path type must be str but you are {}! please correct it!".format(type(img_path)))
        if not exists(img_path):
            raise FileExistsError('image path is incorrect: {}'.format(img_path))
        self.background_frame_path = img_path

    def __call__(self,result:dict):
        """

        Args:
            result (dict): output of model predict.

        Returns:
            result_frame (np.ndarray): treated frame.
        """
        #step1 : read img
        ori_frame = result['frame']
        bcakground_frame = cv2.imread(self.background_frame_path)

        #step2 : accrouding to the ori img shape to resize the background img.
        bcakground_frame = cv2.resize(bcakground_frame, (ori_frame.shape[1], ori_frame.shape[0]), interpolation=cv2.INTER_AREA)

        #step3 : get all mask from segmentation model 
        masks = result["detections"]
        
        #step4 : accrounding to the label to create masks
        

        background_mask, object_mask = self.create_mask(masks)
        

        #step5 : accrounding to the mask to fill the img.

        background_img = cv2.bitwise_and(ori_frame,ori_frame, mask = background_mask )
        object_img = cv2.bitwise_and(bcakground_frame,bcakground_frame, mask = object_mask)

        #step6 : merage background_img and object_mask
        result_frame = cv2.add(background_img,object_img)

        return result_frame


if __name__=='__main__':
    import logging as log
    import cv2
    import numpy as np
    from argparse import ArgumentParser, SUPPRESS
    from ivit_i.io import Source, Displayer
    from ivit_i.core.models import iSegmentation
    from ivit_i.common import Metric

    def build_argparser():

        parser = ArgumentParser(add_help=False)

        basic_args = parser.add_argument_group('Basic options')
        basic_args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
        basic_args.add_argument('-m', '--model', required=True, help='the path to model')
        basic_args.add_argument('-i', '--input', required=True,
                        help='Required. An input to process. The input must be a single image, '
                            'a folder of images, video file or camera id.')
        basic_args.add_argument('-l', '--label', help='Optional. Labels mapping file.', default=None, type=str)
        basic_args.add_argument('-d', '--device', type=str,
                        help='Optional. `Intel` support [ `CPU`, `GPU` ] \
                                `Hailo` is support [ `HAILO` ]; \
                                `Xilinx` support [ `DPU` ]; \
                                `dGPU` support [ 0, ... ] which depends on the device index of your GPUs; \
                                `Jetson` support [ 0 ].' )
        
        model_args = parser.add_argument_group('Model options')
        model_args.add_argument('-t', '--confidence_threshold', default=0.1, type=float,
                                    help='Optional. Confidence threshold for detections.')
        model_args.add_argument('-topk', help='Optional. Number of top results. Default value is 5. Must be from 1 to 10.', default=5,
                                    type=int, choices=range(1, 11))

        io_args = parser.add_argument_group('Input/output options')
        io_args.add_argument('-n', '--name', default='ivit', 
                            help="Optional. The window name and rtsp namespace.")
        io_args.add_argument('-r', '--resolution', type=str, default=None, 
                            help="Optional. Only support usb camera. The resolution you want to get from source object.")
        io_args.add_argument('-f', '--fps', type=int, default=None,
                            help="Optional. Only support usb camera. The fps you want to setup.")
        io_args.add_argument('--no_show', action='store_true',
                            help="Optional. Don't display any stream.")

        args = parser.parse_args()
        # Parse Resoltion
        if args.resolution:
            args.resolution = tuple(map(int, args.resolution.split('x')))

        return args
    

    # 1. Argparse
    args = build_argparser()

    # 2. Basic Parameters
    infer_metrx = Metric()
    
    # 3. Init Model
    model = iSegmentation(
        model_path = args.model,
        label_path = args.label,
        device=args.device
        )
    
    # 4. Init Source
    src = Source(   
        input = args.input, 
        resolution = args.resolution, 
        fps = args.fps )
    
    # 5. Init Display
    if not args.no_show:
        dpr = Displayer( cv = True )

    # 6. Setting iApp
    app_config =   {
                        "application": {
                            "background_frame":"/workspace/data/background.jpg",
                            "change_object":[]
                        }
                    }
    
    app = Change_background(app_config,args.label)
    # 7. Start Inference
    try:
        while(True):
            # Get frame & Do infernece
            frame = src.read()       
            result = model.inference( frame )

            if args.no_show:
                pass
            else:
                # Draw results 
                frame = app(result)              
                infer_metrx.paint_metrics(frame)
                
                # Display
                dpr.show(frame=frame)                   
                if dpr.get_press_key()==ord('q'):
                    break

            # Update Metrix
            infer_metrx.update()

    except KeyboardInterrupt: 
        log.info('Detected Key Interrupt !')

    finally:
        model.release()     # Release Model
        src.release()       # Release Source
        if not args.no_show: 
            dpr.release()   # Release Display
