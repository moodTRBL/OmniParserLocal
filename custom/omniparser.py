from PIL import Image
from typing import List, Tuple, Optional, Dict, Any

from service import UIElementExtractor, ContentParser, ImageProcessor
from omni_dto import ParseConfig, Cordinate

class OmniParser:
    
    def __init__(self, use_paddleocr: bool = False):
        self.image_processor = ImageProcessor()
        self.content_parser = ContentParser(self.ocr_processor, self.image_processor)
        self.ui_extractor = UIElementExtractor(self.content_parser, self.image_processor)
    
    def parse(self, image_source: Image.Image, yolo_model, box_threshold: float, draw_bbox_config, caption_model_processor, iou_threshold: float, use_paddleocr: bool, imgsz: int) -> List[Dict[str, Any]]:
        """
        이미지를 파싱하여 UI 요소들을 추출
        """
        config = ParseConfig(
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            use_paddleocr=use_paddleocr,
            imgsz=imgsz
        )
        
        return self.content_parser.parse_image_content(
            image_source, 
            yolo_model, 
            draw_bbox_config, 
            caption_model_processor, 
            config
        )
    
    def get_clickable_ui(self, parsed_content_list: List[Dict[str, Any]], image_source: Image.Image, width: int, height: int, yolo_model, box_threshold: float, draw_bbox_config, caption_model_processor, iou_threshold: float, use_paddleocr: bool, imgsz: int) -> dict:
        """
        클릭 가능한 UI 요소들을 추출
        """
        config = ParseConfig(
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            use_paddleocr=use_paddleocr,
            imgsz=imgsz
        )
        
        return self.ui_extractor.extract_clickable_ui(
            parsed_content_list, 
            image_source,
            width, 
            height,
            yolo_model, 
            draw_bbox_config, 
            caption_model_processor, 
            config
        )