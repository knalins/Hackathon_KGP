"""
Text Preprocessing Module for BDH NLI Pipeline

Handles:
- Unicode normalization (NFKC)
- Whitespace normalization
- Preserves capitalization and punctuation
- Structured input formatting with metadata
- Empty caption handling
"""

import unicodedata
import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class TextNormalizer:
    """
    Text normalizer following the problem constraints:
    - NFKC Unicode normalization
    - Whitespace normalization
    - NO lowercasing
    - NO stemming/lemmatization
    - Preserve punctuation
    """
    
    def normalize(self, text: str) -> str:
        """
        Normalize text while preserving important features.
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: NFKC Unicode normalization
        # This handles compatibility characters (e.g., ﬁ -> fi)
        text = unicodedata.normalize('NFKC', text)
        
        # Step 2: Normalize various unicode whitespace to regular space
        # But preserve newlines for structure
        text = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000]', ' ', text)
        
        # Step 3: Collapse multiple spaces to single space
        text = re.sub(r' +', ' ', text)
        
        # Step 4: Strip leading/trailing whitespace
        text = text.strip()
        
        # Step 5: Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Step 6: Remove control characters except newline and tab
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        return text
    
    def normalize_novel(self, text: str) -> str:
        """
        Normalize novel text with additional processing.
        
        Args:
            text: Raw novel text
            
        Returns:
            Normalized novel text
        """
        text = self.normalize(text)
        
        # Remove Project Gutenberg headers/footers if present
        start_markers = [
            '*** START OF THE PROJECT GUTENBERG EBOOK',
            '***START OF THE PROJECT GUTENBERG EBOOK',
        ]
        end_markers = [
            '*** END OF THE PROJECT GUTENBERG EBOOK',
            '***END OF THE PROJECT GUTENBERG EBOOK',
            'End of the Project Gutenberg EBook',
            'End of Project Gutenberg',
        ]
        
        # Find start of actual content
        for marker in start_markers:
            if marker in text:
                idx = text.find(marker)
                # Find next line after marker
                newline_idx = text.find('\n', idx)
                if newline_idx != -1:
                    text = text[newline_idx + 1:]
                break
        
        # Find end of actual content
        for marker in end_markers:
            if marker in text:
                idx = text.find(marker)
                text = text[:idx]
                break
        
        return text.strip()


# Special token for empty captions
NO_CAPTION_TOKEN = "[NO_CAPTION]"


def format_backstory_input(
    book_name: str,
    character: str,
    caption: Optional[str],
    content: str,
    normalizer: Optional[TextNormalizer] = None
) -> str:
    """
    Format a backstory sample into structured input.
    
    Format:
    [BOOK: <book_name>]
    [CHARACTER: <character>]
    [CAPTION: <caption or [NO_CAPTION]>]
    [BACKSTORY: <content>]
    
    Args:
        book_name: Name of the novel
        character: Character name
        caption: Optional caption/section title
        content: The backstory content
        normalizer: Optional TextNormalizer instance
        
    Returns:
        Formatted string
    """
    if normalizer is None:
        normalizer = TextNormalizer()
    
    # Normalize all fields
    book_name = normalizer.normalize(book_name)
    character = normalizer.normalize(character)
    content = normalizer.normalize(content)
    
    # Handle empty caption
    if caption is None or (isinstance(caption, str) and caption.strip() == ''):
        caption = NO_CAPTION_TOKEN
    elif isinstance(caption, float):  # Handle NaN from pandas
        import math
        if math.isnan(caption):
            caption = NO_CAPTION_TOKEN
        else:
            caption = str(caption)
    else:
        caption = normalizer.normalize(str(caption))
    
    # Build structured format
    formatted = f"[BOOK: {book_name}]\n"
    formatted += f"[CHARACTER: {character}]\n"
    formatted += f"[CAPTION: {caption}]\n"
    formatted += f"[BACKSTORY: {content}]"
    
    return formatted


def parse_formatted_input(formatted_text: str) -> dict:
    """
    Parse a formatted backstory input back into components.
    
    Args:
        formatted_text: Text formatted by format_backstory_input
        
    Returns:
        Dictionary with book_name, character, caption, backstory
    """
    result = {
        'book_name': None,
        'character': None,
        'caption': None,
        'backstory': None
    }
    
    # Parse each field
    patterns = {
        'book_name': r'\[BOOK:\s*(.*?)\]',
        'character': r'\[CHARACTER:\s*(.*?)\]',
        'caption': r'\[CAPTION:\s*(.*?)\]',
        'backstory': r'\[BACKSTORY:\s*(.*?)\]',
    }
    
    for field, pattern in patterns.items():
        match = re.search(pattern, formatted_text, re.DOTALL)
        if match:
            value = match.group(1).strip()
            if field == 'caption' and value == NO_CAPTION_TOKEN:
                value = None
            result[field] = value
    
    return result


# Label encoding
LABEL_TO_ID = {'consistent': 1, 'contradict': 0}
ID_TO_LABEL = {1: 'consistent', 0: 'contradict'}


def encode_label(label: str) -> int:
    """Convert label string to integer."""
    return LABEL_TO_ID[label.lower().strip()]


def decode_label(label_id: int) -> str:
    """Convert label integer back to string."""
    return ID_TO_LABEL[label_id]
