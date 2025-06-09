from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import logging

from omni_dto import Cordinate, UIElement, ParseConfig
from util.utils import check_ocr_box, get_som_labeled_img

logger = logging.getLogger(__name__)

class ImageProcessor:
    
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

class ContentParser:
    
    def __init__(self, use_paddleocr: bool = False):
        self.use_paddleocr = use_paddleocr
        
    def _extract_text_from_image(self, image_source: Image.Image) -> Optional[Tuple[List[str], List[Cordinate]]]:
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
                
            text, ocr_bbox = ocr_bbox_rslt
            bbox_objects = [Cordinate(*bbox) for bbox in ocr_bbox]
            return text, bbox_objects
            
        except Exception as e:
            logger.error(f"OCR 처리 중 오류 발생: {str(e)}")
            return None
    
    def parse_image_content(self, image_source: Image.Image, yolo_model, draw_bbox_config, caption_model_processor, config: ParseConfig) -> List[Dict[str, Any]]:
        """
        이미지에서 콘텐츠를 파싱
        """
        # OCR 처리
        ocr_result = self._extract_text_from_image(image_source)
        if ocr_result is None:
            return []
        
        text, ocr_bbox = ocr_result
        
        try:
            # SOM 라벨링 실행
            dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                image_source,
                yolo_model=yolo_model,
                BOX_TRESHOLD=config.box_threshold,
                output_coord_in_ratio=True,
                ocr_bbox=[bbox.__dict__ for bbox in ocr_bbox],  # BoundingBox 객체를 dict로 변환
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

class UIElementExtractor:
    
    def __init__(self, content_parser: ContentParser, image_processor: ImageProcessor):
        self.content_parser = content_parser
        self.image_processor = image_processor
    
    def _parse_ui_elements(self, parsed_content_list: List[Dict[str, Any]]) -> Tuple[List[UIElement], List[UIElement]]:
        """
        파싱된 콘텐츠에서 아이콘과 텍스트 요소를 분리
        """
        icons = []
        texts = []
        
        for item in parsed_content_list:
            bbox = Cordinate(*item['bbox'])
            element = UIElement(
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
    
    def _find_text_in_icon(self, icon: UIElement, text_elements: List[UIElement]) -> Optional[str]:
        """
        아이콘 내부에 포함된 텍스트를 찾기
        """
        for text_element in text_elements:
            if icon.bbox.contains(text_element.bbox):
                return text_element.content
        return None
    
    def _process_icon_without_text(self, icon: UIElement, image_source: Image.Image, width: int, height: int, yolo_model, draw_bbox_config, caption_model_processor, config: ParseConfig) -> List[str]:
        """
        텍스트가 없는 아이콘을 별도로 처리
        """
        try:
            # 아이콘 영역 추출
            icon_image = self.image_processor.crop_region(
                image_source=image_source, 
                bbox=icon.bbox, 
                width=width, 
                height=height, 
                min_size=config.min_icon_size
            )
            
            # 추출된 이미지 다시 파싱
            additional_content = self.content_parser.parse_image_content(
                icon_image, 
                yolo_model, 
                draw_bbox_config, 
                caption_model_processor, 
                config
            )
            
            if not additional_content:
                return []
            
            # 텍스트 요소만 추출
            text_contents = [item['content'] for item in additional_content if item['type'] == 'text' and not item['interactivity']]
            
            return text_contents
            
        except Exception as e:
            logger.error(f"아이콘 처리 중 오류: {str(e)}")
            return []
    
    def extract_clickable_ui(self, parsed_content_list: List[Dict[str, Any]], image_source: Image.Image, width: int, height: int, yolo_model, draw_bbox_config, caption_model_processor, config: ParseConfig) -> List[Tuple[str, Cordinate]]:
        """
        클릭 가능한 UI 요소들을 추출
        """
        if not parsed_content_list:
            logger.info("파싱된 콘텐츠가 없습니다")
            return []
        
        icons, texts = self._parse_ui_elements(parsed_content_list)
        ui_elements = []
        
        for icon in icons:
            # 아이콘 내부의 텍스트 찾기
            text_content = self._find_text_in_icon(icon, texts)
            
            if text_content:
                ui_elements.append((text_content, icon.bbox))
            else:
                # 텍스트가 없는 경우 별도 처리
                additional_texts = self._process_icon_without_text(
                    icon, 
                    image_source, 
                    width, 
                    height, 
                    yolo_model,
                    draw_bbox_config, 
                    caption_model_processor, 
                    config
                )
                
                for text in additional_texts:
                    ui_elements.append((text, icon.bbox))
        
        return ui_elements