from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import logging
from collections import defaultdict

from custom.dto import Cordinate, ParseConfig, OmniElement, UIElement
from util.utils import check_ocr_box, get_som_labeled_img

logger = logging.getLogger(__name__)

class ImageUtils:
    
    @staticmethod
    def crop_region(image_source: Image.Image, bbox: Cordinate, width: int, height: int, min_size: int = 200) -> Image.Image:
        """
        아이콘 영역의 이미지를 추출하고 크기를 조정
        """
        # 픽셀 좌표로 변환
        pixel_coords = bbox.to_pixel_coordinates(width, height)
        icon_image = image_source.crop(pixel_coords)
        
        # 크기 계산
        icon_width = pixel_coords[2] - pixel_coords[0]
        icon_height = pixel_coords[3] - pixel_coords[1]
        
        # OCR 성능 향상을 위한 크기 조정
        if icon_width < min_size or icon_height < min_size:
            scale = max(min_size / icon_width, min_size / icon_height)
            new_width = int(icon_width * scale)
            new_height = int(icon_height * scale)
            icon_image = icon_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return icon_image

class ImageParser:
    
    def __init__(self, use_paddleocr: bool = False):
        self.use_paddleocr = use_paddleocr
        
    def _extract_content(self, image_source: Image.Image) -> tuple[list | list[Any | str]]:
        """
        이미지에서 텍스트와 바운딩 박스를 추출
        """
        try:
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
                image_source, 
                display_img=False, 
                output_bb_format='xyxy', 
                goal_filtering=None, 
                easyocr_args={'paragraph': False, 'text_threshold': 0.5}, 
                use_paddleocr=self.use_paddleocr
            )
            
            if ocr_bbox_rslt is None:
                return None
                
            return ocr_bbox_rslt
            
        except Exception as e:
            logger.error(f"OCR 처리 중 오류 발생: {str(e)}")
            return None
    
    def parse_image(self, image_source: Image.Image, yolo_model, draw_bbox_config, caption_model_processor, config: ParseConfig) -> List[Dict[str, Any]]:
        """
        이미지에서 콘텐츠를 파싱
        """
        # OCR 처리
        ocr_result = self._extract_content(image_source)
        if ocr_result is None:
            return []
        
        text, ocr_bbox = ocr_result
        
        try:
            # SOM 라벨링 실행
            dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                image_source,
                model=yolo_model,
                BOX_TRESHOLD=config.box_threshold,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,  # BoundingBox 객체를 dict로 변환
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=caption_model_processor,
                ocr_text=text,
                iou_threshold=config.iou_threshold,
                imgsz=config.imgsz
            )
            
            return parsed_content_list
            
        except Exception as e:
            logger.error(f"콘텐츠 파싱 중 오류 발생: {str(e)}")
            return []        

class UIExtractor:
    
    def __init__(self, image_parser: ImageParser, image_utils: ImageUtils):
        self.image_parser = image_parser
        self.image_utils = image_utils
    
    def _parse_ui_elements(self, parsed_content_list: List[Dict[str, Any]]) -> Tuple[List[OmniElement], List[OmniElement]]:
        """
        파싱된 콘텐츠에서 아이콘과 텍스트 요소를 분리
        """
        icons = []
        texts = []
        
        for item in parsed_content_list:
            bbox = Cordinate(*item['bbox'])
            element = OmniElement(
                content=item['content'],
                bbox=bbox,
                element_type=item['type'],
                is_interactive=item['interactivity']
            )
            
            if element.element_type == 'icon' and element.is_interactive:
                icons.append(element)
            elif element.element_type == 'text' and not element.is_interactive:
                texts.append(element)
        
        return icons, texts
    
    def _combine_ui_list(self, content_list: List[Tuple[str, Cordinate]]) -> list[UIElement]:
        """
        같은 좌표의 텍스트를 하나의 UI로 구성
        """
        ui_list = defaultdict(list)
        for i, (text1, cord1) in enumerate(content_list):
            ui_list[cord1].append(text1)
            for j, (text2, cord2) in enumerate(content_list):
                if i >= j:
                    continue
                
                if cord1.equals(cord2):
                    ui_list[cord1].append(text2)
        
        ui_elements = []
        for cord, contents in ui_list.items():
            ui_elements.append(UIElement(cord, contents))
        
        return ui_elements    
    
    def _remove_duplicate(self, content_list: list[UIElement]) -> list[UIElement]:
        """
        value에 중복되는 텍스트를 제거
        """
        for ui_element in content_list:
            set_list = set(ui_element.contents)
            remove_list = list(set_list)
            ui_element.contents = remove_list

        return content_list
            
    def get_clickable_ui(self, image_source: Image.Image, width: int, height: int, yolo_model, draw_bbox_config, caption_model_processor, config: ParseConfig) -> list[UIElement]:
        """
        클릭 가능한 UI 요소들을 추출
        """ 
        # 이미지 파싱
        parsed_content_list = self.image_parser.parse_image(
            image_source=image_source,
            yolo_model=yolo_model,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=caption_model_processor,
            config=config
        )
        
        icons, texts = self._parse_ui_elements(parsed_content_list)
        ui_elements = []
        #clickable_ui = defaultdict(list)
        
        for icon in icons:
            # 아이콘 내부의 텍스트 찾기
            flag = False
            for text in texts:
                if icon.bbox.contains(text.bbox):
                    ui_elements.append((text.content, icon.bbox))
                    flag = True
            
            if not flag:
                # icon내부에 text가 없다고 판단되는 경우 파싱 한번더
                try:
                    # 아이콘 영역 추출
                    icon_image = self.image_utils.crop_region(
                        image_source=image_source, 
                        bbox=icon.bbox, 
                        width=width, 
                        height=height, 
                        min_size=config.min_icon_size
                    )
                    
                    # 추출된 이미지 다시 파싱
                    additional_content = self.image_parser.parse_image(
                        icon_image, 
                        yolo_model, 
                        draw_bbox_config, 
                        caption_model_processor, 
                        config
                    )
                    
                    if not additional_content:
                        continue
                    
                    # 텍스트 요소만 추출
                    additional_texts = [item['content'] for item in additional_content if item['type'] == 'text' and not item['interactivity']]                    
                except Exception as e:
                    logger.error(f"아이콘 처리 중 오류: {str(e)}")
                    continue
                
                for text in additional_texts:
                    ui_elements.append((text, icon.bbox))
        
        clickable_ui = self._combine_ui_list(ui_elements)
        clickable_ui = self._remove_duplicate(clickable_ui)
        
        return clickable_ui