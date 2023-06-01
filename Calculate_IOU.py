#!/usr/bin/env python3
# Copyright (c) 2023 Innodisk Corporation
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os, cv2, logging
import numpy as np
from apps.palette import palette
from os.path import exists
from typing import Union, get_args
from ivit_i.common.app import iAPP_SEG

class Calculate_IOU(iAPP_SEG):
    def __init__(self,params:dict=None, label_path:str=None, palette=palette):
        """
        Args:
            params (dict, optional): _description_. Defaults to None.
            label (str, optional): _description_. Defaults to None.
        """
        self.params = params
        self.label = self._check_label(label_path)
        self.depend_on = self._check_depend_on()

        #for draw
        self.palette = self._init_palette(palette)
        self.draw_mask = True
        self.draw_iou_result = True

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

    def _check_depend_on(self):
        """
        step1 : Verify app config.
        step2 : Verify depend_on.
        If all right , return a list that contain objcet user want to calculate iou.

        Returns:
            list: Return a list that contain objcet user want to calculate iou. 
        """
        if not self.params['application'].__contains__('depend_on'):
            raise KeyError("App config not set change object! please correct it!")
        
        if not isinstance(self.params['application']['depend_on'],list):
            raise ValueError("Depend on type must be list but you set {}! please correct it!".format(type(self.params['application']['depend_on'])))
        
        if len(self.params['application']['depend_on'])==0:
            return self.label
        else:
            return self.params['application']['depend_on']
        
    def calculate_iou(self,masks:np.ndarray):
        """
        step1 : Check depend_on whether in label or not.
        step2 : Calculate iou.
 
        Args:
            masks (np.ndarray): output of model predict , shape must same with input image.

        Returns:
            iou (dict): Iou of objcet in whole frame.
        """
       
        iou={}
        for label in self.depend_on:
            if not (label in self.label):
                raise ValueError(" {} not in label!".format(label))
            iou.update({label:np.sum(masks==self.label.index(label))/(masks.shape[0]*masks.shape[1])})
        
        
        return iou

    def change_depend_on(self,new_depend_on:str):
        """
        If you want to change depend_on you can use this function.

        Args:
            img_path (str): New depend_on.

        """
        if isinstance(new_depend_on,list):
            raise ValueError("The new_depend_on type must be lsit but you are {}! please correct it!".format(type(new_depend_on)))
        if len(new_depend_on)==0:
            raise ValueError('The new_depend_on: {} is empty , please correct it!'.format(new_depend_on))
        self.depend_on = new_depend_on

    def _init_palette(self,palette):
        """
        Init color for each object.
        Args:
            palette (dict): Defalt pallete zoo.
        Returns:
            (dict): Each label have one color. 
        """
        temp_palette = {}
        for idx , label in enumerate(self.label):
            if self.params['application'].__contains__('palette'):
                if self.params['application']['palette'].__contains__(label):
                    temp_palette.update({label:self.params['application']['palette'][label]})
                else:
                    temp_palette.update({label:palette[str(idx+1)]})
            else:
                temp_palette.update({label:palette[str(idx+1)]})
        return temp_palette
    
    def draw(self,result:dict,iou_result:dict,draw_mask:bool=True,draw_iou_result:bool=True):
        
        #step1 : Get ori frame and mask.
        frame = result['frame']
        masks_B = result['detections'].copy()
        masks_G = result['detections'].copy()
        masks_R = result['detections'].copy()

        #step2 : according to the pallete change the mask color.
        for idx,label in enumerate(self.label):
            masks_B[masks_B==idx]=self.palette[label][0]
            masks_G[masks_G==idx]=self.palette[label][1]
            masks_R[masks_R==idx]=self.palette[label][2]

        #step3 : merage masks with 3 chnnel.
        masks_3d = cv2.merge([masks_B, masks_G, masks_R])
        
        #step4 : draw the mask on the ori frame.
        draw_mask = self.draw_mask
        if draw_mask:
            frame = np.floor_divide(frame, 2) + np.floor_divide(masks_3d, 2)
        
        #step5 : draw result of iou 
        draw_iou_result = self.draw_iou_result
        if draw_iou_result:

            #Draw param
            font_scale = 0.7 # float 
            font_face = cv2.FONT_HERSHEY_COMPLEX #int
            font_thick = 2 #int
            default_color = (255, 0, 0) #tuple
            idx=0

            # Initialize Position
            first_label_name = self.label[0] if self.label else ""
            label_height = cv2.getTextSize(first_label_name, font_face, font_scale, 2)[0][1]
            initial_labels_pos =  frame.shape[0] - label_height * (int(1.5 * len(self.label)) + 1)

            if (initial_labels_pos < 0):
                initial_labels_pos = label_height
                log.warning('Too much labels to display on this frame, some will be omitted')
            offset_y = initial_labels_pos
            # Draw Header
            header = "Label:     IOU:"
            label_width = cv2.getTextSize(header, font_face, font_scale, font_thick)[0][0]

            cv2.putText(frame, header, (frame.shape[1] - label_width, offset_y), font_face, font_scale, (255, 255, 255), font_thick + 1) # white border
            cv2.putText(frame, header, (frame.shape[1] - label_width, offset_y), font_face, font_scale, default_color, font_thick)
            
            # Draw Detections
            for label_name, label_score in iou_result.items():
                
                # Get Color
                color = self.palette[label_name] if self.palette else default_color
                
                # Get Label Content
                label = '{}. {}    {:.2f}'.format(idx, label_name, label_score)
                label_width = cv2.getTextSize(label, font_face, font_scale, font_thick)[0][0]

                # Draw Label Content
                offset_y += int(label_height * 1.5)
                cv2.putText(frame, label, (frame.shape[1] - label_width, offset_y), font_face, font_scale, (255, 255, 255), font_thick + 1) # white border
                cv2.putText(frame, label, (frame.shape[1] - label_width, offset_y), font_face, font_scale, color, font_thick)
                idx+=1

        return frame
    
    def set_draw(self,params:dict):
        """
        Control anything about drawing.
        Which params you can contral :

        { 
            draw_mask : bool , 
            draw_iou_result : bool ,
            palette (dict) { label(str) : color(Union[tuple, list]) }
        }
        
        Args:
            params (dict): 
        """
        if isinstance(params.get('draw_mask', self.draw_mask) , bool):    
            self.draw_mask= params.get('draw_mask', self.draw_mask)
            logging.info("Change draw_mask mode , now draw_mask mode is {} !".format(self.draw_mask))
        else:
            logging.error("draw_mask type is bool! but your type is {} ,please correct it.".format(type(params.get('draw_mask', self.draw_mask))))

        if isinstance(params.get('draw_iou_result', self.draw_iou_result) , bool):    
            self.draw_iou_result= params.get('draw_iou_result', self.draw_iou_result)
            logging.info("Change draw_iou_result mode , now draw_iou_result mode is {} !".format(self.draw_iou_result))
        else:
            logging.error("draw_iou_result type is bool! but your type is {} ,please correct it.".format(type(params.get('draw_iou_result', self.draw_iou_result))))
        
        color_support_type = Union[tuple, list]
        palette = params.get('palette', None)
        if isinstance(palette, dict):
            if len(palette)==0:
                logging.warning("Not set palette!")
                pass
            else:
                for label,color in palette.items():

                    if isinstance(label, str) and isinstance(color, get_args(color_support_type)):
                        if self.palette.__contains__(label):
                           self.palette.update({label:color})
                        else:
                            logging.error("Model can't recognition the label {} , please checkout your label!.".format(label))
                        logging.info("Label: {} , change color to {}.".format(label,color))
                    else:
                        logging.error("Value in palette type must (label:str , color :Union[tuple , list] ),your type \
                                      label:{} , color:{} is error.".format(type(label),type(color)))
        else:
            logging.error("Not set palette or your type {} is error.".format(type(palette)))

    def __call__(self,result:dict):
        """

        Args:
            result (dict): output of model predict.

        Returns:
            result_frame (np.ndarray): treated frame.
        """

        #step1 : get all mask from segmentation model 
        masks = result["detections"]
        
        #step2 : accrounding to the label to calculate iou.
        
        iou_result = self.calculate_iou(masks)

        #step3 : draw the result on the image

        frame = self.draw(result,iou_result)

        return frame , iou_result 


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
                            "depend_on":[],
                            "palette":{
                                "car":[0,0,0]

                            }
                        }
                    }
    
    app = Calculate_IOU(app_config,args.label)
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
                frame,info = app(result)    
                # a={ 
                #     'draw_mask' : True , 
                #     'draw_iou_result' : True 
                #     # "palette" :{ "car" : [0,0,0] },
                # }   
                # app.set_draw(a)
                # print(info)          
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
