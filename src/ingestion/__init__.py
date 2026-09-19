"""
Ingestion Module

Dosya parse etme ve içerik çıkarma modülü.
"""

from src.ingestion.base import BaseParser, ParserResult, ParserError
from src.ingestion.text import TextParser, TXTParser, MDParser
from src.ingestion.pdf import PDFParser, SecurePDFParser

__all__ = [
    "BaseParser",
    "ParserResult",
    "ParserError",
    "TextParser",
    "TXTParser",
    "MDParser",
    "PDFParser",
    "SecurePDFParser",
]


# Parser factory
def get_parser(file_extension: str) -> BaseParser:
    """
    Dosya extension'ına göre uygun parser'ı döndür.
    
    Args:
        file_extension: Dosya extension'ı (örn: '.txt', '.pdf')
        
    Returns:
        Parser instance
        
    Raises:
        ValueError: Desteklenmeyen extension
    """
    extension = file_extension.lower()
    
    parser_map = {
        '.txt': TXTParser,
        '.text': TXTParser,
        '.md': MDParser,
        '.markdown': MDParser,
        '.pdf': PDFParser,
    }
    
    parser_class = parser_map.get(extension)
    
    if parser_class is None:
        raise ValueError(f"Unsupported file extension: {extension}")
    
    return parser_class()
