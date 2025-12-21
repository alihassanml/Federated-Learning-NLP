
import pdfplumber
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import pandas as pd
import io
import os
import json
import numpy as np
from typing import List, Dict, Tuple
import cv2

class MultimodalPDFParser:
    # \"\"\"
    # Advanced PDF parser supporting:
    # - Text extraction
    # - Table detection and extraction
    # - Image extraction with OCR
    # - Chart/figure detection
    # - Layout-aware parsing
    # \"\"\"
    
    def __init__(self, tesseract_path: str = None):
        # Initialize parser
        
        # Args:
        #     tesseract_path: Path to tesseract executable (if not in PATH)
        #                    Windows: 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        #                    Linux: usually in PATH
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.supported_formats = ['.pdf']
        
    def parse_pdf(self, pdf_path: str, extract_images: bool = True, 
                  extract_tables: bool = True, ocr_images: bool = True) -> Dict:
        # \"\"\"
        # Parse PDF with multimodal content extraction
        
        # Args:
        #     pdf_path: Path to PDF file
        #     extract_images: Whether to extract images
        #     extract_tables: Whether to extract tables
        #     ocr_images: Whether to run OCR on images
            
        # Returns:
        #     Dict with extracted content: {
        #         'text': str,
        #         'tables': List[pd.DataFrame],
        #         'images': List[Dict],
        #         'metadata': Dict
        #     }
        # \"\"\"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        result = {
            'text': '',
            'tables': [],
            'images': [],
            'metadata': {},
            'pages': []
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                result['metadata'] = {
                    'num_pages': len(pdf.pages),
                    'filename': os.path.basename(pdf_path),
                    'file_size': os.path.getsize(pdf_path)
                }
                
                # Process each page
                for page_num, page in enumerate(pdf.pages, 1):
                    page_content = self._process_page(
                        page, page_num, extract_images, extract_tables
                    )
                    
                    result['pages'].append(page_content)
                    result['text'] += page_content['text'] + '\n\n'
                    result['tables'].extend(page_content['tables'])
                    result['images'].extend(page_content['images'])
            
            # OCR on extracted images if requested
            if ocr_images and extract_images:
                result = self._ocr_images(result, pdf_path)
            
            print(f"✓ Parsed {pdf_path}:")
            print(f"  - Text: {len(result['text'])} characters")
            print(f"  - Tables: {len(result['tables'])}")
            print(f"  - Images: {len(result['images'])}")
            
            return result
            
        except Exception as e:
            print(f"Error parsing PDF {pdf_path}: {e}")
            return result
    
    def _process_page(self, page, page_num: int, extract_images: bool, 
                     extract_tables: bool) -> Dict:
        # \"\"\"Process a single PDF page\"\"\"
        page_content = {
            'page_number': page_num,
            'text': '',
            'tables': [],
            'images': []
        }
        
        # Extract text
        text = page.extract_text()
        if text:
            page_content['text'] = text
        
        # Extract tables
        if extract_tables:
            tables = page.extract_tables()
            for table_idx, table in enumerate(tables):
                if table:
                    try:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        page_content['tables'].append({
                            'page': page_num,
                            'table_index': table_idx,
                            'dataframe': df,
                            'text_representation': self._table_to_text(df)
                        })
                    except Exception as e:
                        print(f"  Warning: Could not parse table on page {page_num}: {e}")
        
        # Extract images
        if extract_images:
            images = page.images
            for img_idx, img in enumerate(images):
                page_content['images'].append({
                    'page': page_num,
                    'image_index': img_idx,
                    'bbox': (img['x0'], img['top'], img['x1'], img['bottom']),
                    'width': img['width'],
                    'height': img['height']
                })
        
        return page_content
    
    def _table_to_text(self, df: pd.DataFrame) -> str:
        # \"\"\"Convert DataFrame to readable text\"\"\"
        try:
            # Create readable table representation
            text = f"Table with {len(df)} rows and {len(df.columns)} columns:\n"
            text += "Columns: " + ", ".join(str(col) for col in df.columns) + "\n"
            
            # Add sample rows (first 3)
            for idx, row in df.head(3).iterrows():
                row_text = " | ".join(str(val) for val in row.values)
                text += f"Row {idx+1}: {row_text}\n"
            
            if len(df) > 3:
                text += f"... and {len(df) - 3} more rows\n"
            
            return text
        except Exception as e:
            return f"Table (could not convert to text: {e})"
    
    def _ocr_images(self, result: Dict, pdf_path: str) -> Dict:
        # \"\"\"Run OCR on extracted images\"\"\"
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path, dpi=200)
            
            for img_info in result['images']:
                page_num = img_info['page']
                if page_num <= len(images):
                    page_img = images[page_num - 1]
                    
                    # Crop to image bbox
                    bbox = img_info['bbox']
                    cropped = page_img.crop(bbox)
                    
                    # Run OCR
                    try:
                        ocr_text = pytesseract.image_to_string(cropped)
                        img_info['ocr_text'] = ocr_text.strip()
                        
                        # Add OCR text to main text
                        if ocr_text.strip():
                            result['text'] += f"\n[Image OCR]: {ocr_text}\n"
                    except Exception as e:
                        img_info['ocr_text'] = f"OCR failed: {e}"
            
        except Exception as e:
            print(f"  Warning: Image OCR failed: {e}")
        
        return result
    
    def extract_to_chunks(self, parsed_result: Dict, chunk_size: int = 500, 
                         overlap: int = 50, include_tables: bool = True) -> List[Dict]:
        # \"\"\"
        # Convert parsed multimodal content into chunks for RAG
        
        # Args:
        #     parsed_result: Output from parse_pdf()
        #     chunk_size: Characters per chunk
        #     overlap: Overlap between chunks
        #     include_tables: Whether to include table text in chunks
            
        # Returns:
        #     List of chunks with metadata
        # \"\"\"
        chunks = []
        
        # Combine all text
        full_text = parsed_result['text']
        
        # Add table text if requested
        if include_tables:
            for table_info in parsed_result['tables']:
                full_text += f"\n\n{table_info['text_representation']}\n\n"
        
        # Create text chunks
        start = 0
        chunk_id = 0
        
        while start < len(full_text):
            end = start + chunk_size
            chunk_text = full_text[start:end]
            
            # Try to break at sentence boundary
            if end < len(full_text):
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text.strip(),
                'start_char': start,
                'end_char': end,
                'source': parsed_result['metadata']['filename'],
                'type': 'multimodal'
            })
            
            chunk_id += 1
            start = end - overlap
        
        return chunks
