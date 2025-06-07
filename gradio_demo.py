from typing import Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image
import io


import base64, os
from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import torch
from PIL import Image
import json

yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
# caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2")

MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

DEVICE = torch.device('cuda')

# @spaces.GPU
# @torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process(
    image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz
) -> Optional[Image.Image]:

    box_overlay_ratio = image_input.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    # import pdb; pdb.set_trace()

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_input, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_input, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz,)  
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print('finish processing')
    
    w, h = image_input.size
    ui_list = get_clickable_ui(parsed_content_list, image_input, w, h, yolo_model, box_threshold, draw_bbox_config, caption_model_processor, iou_threshold, use_paddleocr, imgsz)
    for key, value in ui_list:
         print(f'{key}: {value}')
    
    parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
    
    return image, str(parsed_content_list)
    

#상호작용 가능한 UI만 추출
def get_clickable_ui(parsed_content_list, image_source, w, h, yolo_model, box_threshold, draw_bbox_config, caption_model_processor, iou_threshold, use_paddleocr, imgsz):
    icon_list = [(value['bbox'], value['content']) for value in parsed_content_list if value['type'] == 'icon' and value['interactivity'] == True]
    text_list = [(value['bbox'], value['content']) for value in parsed_content_list if value['type'] == 'text' and value['interactivity'] == False]
    ui_list = []
    
    for icon_cord, icon_contnet in icon_list:
        flag = False
        for text_cord, text_content in text_list:
            icon_mnx, icon_mny, icon_mxx, icon_mxy = icon_cord
            text_mnx, text_mny, text_mxx, text_mxy = text_cord
            if icon_mnx < text_mnx and icon_mny < text_mny and icon_mxx > text_mxx and icon_mxy > text_mxy:
                ui_list.append((text_content, icon_cord))
                flag = True
        
        #텍스트가 제대로 인식되지 않은 UI만 따로 추출하여 OmniParser를 실행
        if not flag:
            # 아이콘 영역의 이미지를 추출
            icon_mnx, icon_mny, icon_mxx, icon_mxy = icon_cord
            icon_width = int((icon_mxx - icon_mnx) * w)
            icon_height = int((icon_mxy - icon_mny) * h)
            
            # 원본 해상도로 이미지 추출
            icon_image = image_source.crop((int(icon_mnx * w), int(icon_mny * h), int(icon_mxx * w), int(icon_mxy * h)))
            
            # OCR이 잘 동작하도록 이미지 크기 조정 
            min_size = 200
            if icon_width < min_size or icon_height < min_size:
                scale = max(min_size / icon_width, min_size / icon_height)
                new_width = int(icon_width * scale)
                new_height = int(icon_height * scale)
                icon_image = icon_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 추출된 이미지로 OmniParser 다시 실행
            try:
                ocr_bbox_rslt, is_goal_filtered = check_ocr_box(icon_image, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.5}, use_paddleocr=use_paddleocr)
                if ocr_bbox_rslt is None:
                    continue
                text, ocr_bbox = ocr_bbox_rslt
            except:
                continue
            
            dino_labled_img, label_coordinates, parsed_content_list_second = get_som_labeled_img(
                icon_image, 
                yolo_model, 
                BOX_TRESHOLD=box_threshold, 
                output_coord_in_ratio=True, 
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config, 
                caption_model_processor=caption_model_processor,
                ocr_text=text,
                iou_threshold=iou_threshold, 
                imgsz=imgsz
            )
            
            # image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
            # image.save(r"C:\Users\hachi\PythonProjects\file-save\icon.png")
        
            text_list_second = [(value['bbox'], value['content']) for value in parsed_content_list_second if value['type'] == 'text' and value['interactivity'] == False]
            print(text_list_second)
            
             # 새로운 결과에서 텍스트가 있는지 확인
            for text_cord, text_content in text_list_second:
                ui_list.append((text_content, icon_cord))
            
        
    return ui_list
                

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image_input_component = gr.Image(
                type='pil', label='Upload image')
            # set the threshold for removing the bounding boxes with low confidence, default is 0.05
            box_threshold_component = gr.Slider(
                label='Box Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.05)
            # set the threshold for removing the bounding boxes with large overlap, default is 0.1
            iou_threshold_component = gr.Slider(
                label='IOU Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.1)
            use_paddleocr_component = gr.Checkbox(
                label='Use PaddleOCR', value=True)
            imgsz_component = gr.Slider(
                label='Icon Detect Image Size', minimum=640, maximum=1920, step=32, value=640)
            submit_button_component = gr.Button(
                value='Submit', variant='primary')
        with gr.Column():
            image_output_component = gr.Image(type='pil', label='Image Output')
            text_output_component = gr.Textbox(label='Parsed screen elements', placeholder='Text Output')

    submit_button_component.click(
        fn=process,
        inputs=[
            image_input_component,
            box_threshold_component,
            iou_threshold_component,
            use_paddleocr_component,
            imgsz_component
        ],
        outputs=[image_output_component, text_output_component]
    )

# demo.launch(debug=False, show_error=True, share=True)
demo.launch(share=True, server_port=7861, server_name='0.0.0.0')
