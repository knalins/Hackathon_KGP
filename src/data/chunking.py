"""
Novel Chunking Module for BDH NLI Pipeline

Implements sentence-boundary-aware chunking with overlap
to handle long novels without truncation.
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ChunkConfig:
    """Configuration for novel chunking."""
    
    # Target size for each chunk (in characters, approximates tokens)
    target_chars: int = 1024  # ~256 tokens at ~4 chars/token
    
    # Overlap between consecutive chunks (in characters)
    overlap_chars: int = 256  # ~64 tokens
    
    # Minimum chunk size (don't create tiny chunks)
    min_chunk_chars: int = 512  # ~128 tokens
    
    # Maximum chunk size (hard limit)
    max_chunk_chars: int = 2048  # ~512 tokens
    
    # Sentence boundary patterns (in order of preference)
    sentence_endings: Tuple[str, ...] = (
        '. ', '! ', '? ',  # Standard endings with space
        '."', '!"', '?"',  # Endings before quote
        '.\'', '!\'', '?\'',
        '.\n', '!\n', '?\n',  # Endings at line break
    )


class NovelChunker:
    """
    Chunker that respects sentence boundaries and provides overlap.
    
    Key features:
    - Never breaks mid-sentence
    - Provides configurable overlap for context continuity
    - Handles edge cases (very long sentences, chapter breaks)
    """
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
    
    def _find_sentences(self, text: str) -> List[Tuple[int, int]]:
        """
        Find sentence boundaries in text.
        
        Returns list of (start, end) indices for each sentence.
        """
        sentences = []
        
        # Split on sentence-ending patterns
        # This regex matches sentence endings followed by space/newline or end of string
        pattern = r'[.!?]["\']?\s+'
        
        last_end = 0
        for match in re.finditer(pattern, text):
            end = match.end()
            if end - last_end > 0:
                sentences.append((last_end, end))
            last_end = end
        
        # Add remaining text as final sentence
        if last_end < len(text):
            sentences.append((last_end, len(text)))
        
        return sentences
    
    def _find_paragraph_breaks(self, text: str) -> List[int]:
        """Find paragraph break positions (double newlines)."""
        breaks = []
        for match in re.finditer(r'\n\s*\n', text):
            breaks.append(match.start())
        return breaks
    
    def chunk_text(self, text: str) -> List[dict]:
        """
        Chunk text into overlapping segments respecting sentence boundaries.
        
        Args:
            text: The full novel text (normalized)
            
        Returns:
            List of chunk dictionaries with:
            - 'text': chunk content
            - 'start_char': starting character position
            - 'end_char': ending character position
            - 'chunk_idx': chunk index
        """
        if not text:
            return []
        
        sentences = self._find_sentences(text)
        
        if not sentences:
            # No sentence boundaries found, fall back to character-based chunking
            return self._chunk_by_chars(text)
        
        chunks = []
        current_chunk_start = 0
        current_chunk_sentences = []
        current_chunk_len = 0
        
        for sent_start, sent_end in sentences:
            sent_len = sent_end - sent_start
            
            # Check if adding this sentence would exceed target
            if current_chunk_len + sent_len > self.config.target_chars:
                # Save current chunk if it meets minimum
                if current_chunk_len >= self.config.min_chunk_chars:
                    chunk_text = text[current_chunk_start:current_chunk_sentences[-1][1]]
                    chunks.append({
                        'text': chunk_text,
                        'start_char': current_chunk_start,
                        'end_char': current_chunk_sentences[-1][1],
                        'chunk_idx': len(chunks)
                    })
                    
                    # Find overlap start point
                    overlap_start = self._find_overlap_start(
                        current_chunk_sentences, 
                        current_chunk_start,
                        text
                    )
                    current_chunk_start = overlap_start
                    
                    # Keep sentences that are in overlap region
                    current_chunk_sentences = [
                        (s, e) for s, e in current_chunk_sentences 
                        if s >= overlap_start
                    ]
                    current_chunk_len = sum(e - s for s, e in current_chunk_sentences)
            
            # Add sentence to current chunk
            current_chunk_sentences.append((sent_start, sent_end))
            current_chunk_len += sent_len
            
            # Handle very long sentences (exceed max)
            if sent_len > self.config.max_chunk_chars:
                # Force split the long sentence
                self._add_long_sentence_chunks(
                    text, sent_start, sent_end, chunks
                )
                current_chunk_sentences = []
                current_chunk_len = 0
                current_chunk_start = sent_end
        
        # Add final chunk
        if current_chunk_sentences and current_chunk_len >= self.config.min_chunk_chars:
            chunk_text = text[current_chunk_start:current_chunk_sentences[-1][1]]
            chunks.append({
                'text': chunk_text,
                'start_char': current_chunk_start,
                'end_char': current_chunk_sentences[-1][1],
                'chunk_idx': len(chunks)
            })
        elif current_chunk_sentences:
            # Merge with previous chunk if too small
            if chunks:
                prev_chunk = chunks[-1]
                prev_chunk['text'] = text[prev_chunk['start_char']:current_chunk_sentences[-1][1]]
                prev_chunk['end_char'] = current_chunk_sentences[-1][1]
            else:
                # Only chunk, even if small
                chunk_text = text[current_chunk_start:current_chunk_sentences[-1][1]]
                chunks.append({
                    'text': chunk_text,
                    'start_char': current_chunk_start,
                    'end_char': current_chunk_sentences[-1][1],
                    'chunk_idx': len(chunks)
                })
        
        # Re-index chunks
        for i, chunk in enumerate(chunks):
            chunk['chunk_idx'] = i
        
        return chunks
    
    def _find_overlap_start(
        self, 
        sentences: List[Tuple[int, int]], 
        chunk_start: int,
        text: str
    ) -> int:
        """Find where to start the next chunk for overlap."""
        if not sentences:
            return chunk_start
        
        total_len = sentences[-1][1] - chunk_start
        overlap_target = total_len - self.config.overlap_chars
        
        # Find first sentence that starts after overlap target
        for sent_start, sent_end in sentences:
            if sent_start - chunk_start >= overlap_target:
                return sent_start
        
        # Default to last sentence start
        return sentences[-1][0]
    
    def _add_long_sentence_chunks(
        self, 
        text: str, 
        sent_start: int, 
        sent_end: int, 
        chunks: List[dict]
    ):
        """Handle sentences longer than max_chunk_chars by splitting on clause boundaries."""
        sentence = text[sent_start:sent_end]
        
        # Try to split on clause boundaries
        clause_patterns = ['; ', ', ', ' - ', ': ']
        
        for pattern in clause_patterns:
            if pattern in sentence:
                parts = sentence.split(pattern)
                current_part = ""
                part_start = sent_start
                
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:
                        part = part + pattern
                    
                    if len(current_part) + len(part) > self.config.max_chunk_chars:
                        if current_part:
                            chunks.append({
                                'text': current_part,
                                'start_char': part_start,
                                'end_char': part_start + len(current_part),
                                'chunk_idx': len(chunks)
                            })
                        part_start = part_start + len(current_part)
                        current_part = part
                    else:
                        current_part += part
                
                if current_part:
                    chunks.append({
                        'text': current_part,
                        'start_char': part_start,
                        'end_char': part_start + len(current_part),
                        'chunk_idx': len(chunks)
                    })
                return
        
        # Fallback: split by characters
        for i in range(0, len(sentence), self.config.max_chunk_chars):
            chunk_text = sentence[i:i + self.config.max_chunk_chars]
            chunks.append({
                'text': chunk_text,
                'start_char': sent_start + i,
                'end_char': sent_start + i + len(chunk_text),
                'chunk_idx': len(chunks)
            })
    
    def _chunk_by_chars(self, text: str) -> List[dict]:
        """Fallback character-based chunking when no sentence boundaries found."""
        chunks = []
        pos = 0
        
        while pos < len(text):
            end = min(pos + self.config.target_chars, len(text))
            
            # Try to find a space to break on
            if end < len(text):
                space_pos = text.rfind(' ', pos, end)
                if space_pos > pos:
                    end = space_pos + 1
            
            chunk_text = text[pos:end]
            chunks.append({
                'text': chunk_text,
                'start_char': pos,
                'end_char': end,
                'chunk_idx': len(chunks)
            })
            
            # Advance with overlap
            pos = end - self.config.overlap_chars
            if pos <= chunks[-1]['start_char']:
                pos = end
        
        return chunks
    
    def chunk_novel(self, novel_text: str, book_name: str) -> List[dict]:
        """
        Chunk a full novel and add metadata.
        
        Args:
            novel_text: Full novel text (should be normalized first)
            book_name: Name of the book for metadata
            
        Returns:
            List of chunk dicts with book metadata added
        """
        chunks = self.chunk_text(novel_text)
        
        for chunk in chunks:
            chunk['book_name'] = book_name
        
        return chunks


def chunk_novels(novels: dict, chunker: Optional[NovelChunker] = None) -> dict:
    """
    Chunk multiple novels.
    
    Args:
        novels: Dict mapping book_name -> novel_text
        chunker: Optional NovelChunker instance
        
    Returns:
        Dict mapping book_name -> list of chunks
    """
    if chunker is None:
        chunker = NovelChunker()
    
    chunked_novels = {}
    for book_name, text in novels.items():
        chunks = chunker.chunk_novel(text, book_name)
        chunked_novels[book_name] = chunks
        print(f"Chunked '{book_name}': {len(chunks)} chunks")
    
    return chunked_novels
