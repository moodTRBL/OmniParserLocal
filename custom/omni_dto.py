from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class Cordinate:
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    
    def contains(self, other: 'Cordinate') -> bool:
        """
        다른 바운딩 박스가 이 박스 안에 포함되는지 확인
        """
        return (self.min_x < other.min_x and 
                self.min_y < other.min_y and 
                self.max_x > other.max_x and 
                self.max_y > other.max_y)
        
    def to_pixel_coordinates(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        비율 좌표를 픽셀 좌표로 변환
        """
        return (
            int(self.min_x * width),
            int(self.min_y * height),
            int(self.max_x * width),
            int(self.max_y * height)
        )
    
@dataclass
class UIElement:
    content: str
    bbox: Cordinate
    element_type: str
    is_interactive: bool

@dataclass
class ParseConfig:
    box_threshold: float
    iou_threshold: float
    use_paddleocr: bool
    imgsz: int
    min_icon_size: int = 200