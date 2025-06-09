from PIL import Image

from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

class OmniParser():
    def cut_image(image_source, icon_cord, w, h):
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

        return icon_image        
    
    # OmniParser 실행
    def parse(image_source, yolo_model, box_threshold, draw_bbox_config, caption_model_processor, iou_threshold, use_paddleocr, imgsz):
        try:
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_source, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.5}, use_paddleocr=use_paddleocr)
            if ocr_bbox_rslt is None:
                return []
            text, ocr_bbox = ocr_bbox_rslt
        except:
            return []
                
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_source, 
            yolo_model=yolo_model, 
            BOX_TRESHOLD=box_threshold, 
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=caption_model_processor,
            ocr_text=text,
            iou_threshold=iou_threshold, 
            imgsz=imgsz
        )

        return parsed_content_list

    #상호작용 가능한 UI만 추출
    def get_clickable_ui(self, parsed_content_list, image_source, w, h, yolo_model, box_threshold, draw_bbox_config, caption_model_processor, iou_threshold, use_paddleocr, imgsz):
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
                icon_image = self.cut_image(image_source, icon_cord, w, h)
                parsed_content_list = self.parse(icon_image, yolo_model, box_threshold, draw_bbox_config, caption_model_processor, iou_threshold, use_paddleocr, imgsz)
                if not parsed_content_list:
                    continue
            
                text_list_second = [(value['bbox'], value['content']) for value in parsed_content_list if value['type'] == 'text' and value['interactivity'] == False]
                print(text_list_second)
                
                # 새로운 결과에서 텍스트가 있는지 확인
                for text_cord, text_content in text_list_second:
                    ui_list.append((text_content, icon_cord))
                
            
        return ui_list