"""
Byte-Level Tokenizer for BDH NLI Pipeline

Uses byte-level encoding (vocab_size=256) to handle any text
without vocabulary limitations.
"""

import torch
from typing import List, Union, Optional


class ByteTokenizer:
    """
    Simple byte-level tokenizer.
    
    Converts text to bytes and treats each byte as a token.
    This ensures full coverage of any character without OOV issues.
    
    Vocab size: 256 (one token per byte value)
    """
    
    def __init__(self, max_length: Optional[int] = None, padding: bool = True):
        """
        Args:
            max_length: Maximum sequence length (truncate if exceeded)
            padding: Whether to pad sequences to max_length
        """
        self.max_length = max_length
        self.padding = padding
        self.vocab_size = 256
        
        # Special token IDs (using reserved byte positions)
        self.pad_token_id = 0  # NULL byte for padding
        self.unk_token_id = 1  # Unlikely to appear in text
    
    def encode(
        self, 
        text: str, 
        max_length: Optional[int] = None,
        padding: Optional[bool] = None,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode text to byte tokens.
        
        Args:
            text: Input text to encode
            max_length: Override instance max_length
            padding: Override instance padding
            return_tensors: If 'pt', return PyTorch tensor
            
        Returns:
            List of token IDs or tensor
        """
        max_len = max_length or self.max_length
        do_padding = padding if padding is not None else self.padding
        
        # Convert to bytes
        byte_array = text.encode('utf-8', errors='replace')
        token_ids = list(byte_array)
        
        # Truncate if needed
        if max_len is not None and len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        
        # Pad if needed
        if do_padding and max_len is not None:
            padding_length = max_len - len(token_ids)
            if padding_length > 0:
                token_ids = token_ids + [self.pad_token_id] * padding_length
        
        # Return as tensor if requested
        if return_tensors == 'pt':
            return torch.tensor(token_ids, dtype=torch.long)
        
        return token_ids
    
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List or tensor of token IDs
            
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Remove padding
        token_ids = [t for t in token_ids if t != self.pad_token_id]
        
        # Convert back to bytes and decode
        byte_array = bytes(token_ids)
        text = byte_array.decode('utf-8', errors='replace')
        
        return text
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        return_tensors: Optional[str] = None
    ) -> Union[List[List[int]], torch.Tensor]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            padding: Whether to pad to max length
            return_tensors: If 'pt', return PyTorch tensor
            
        Returns:
            Batch of encoded sequences
        """
        max_len = max_length or self.max_length
        
        # If no max_length specified, use the longest sequence
        if max_len is None:
            encoded = [self.encode(t, padding=False) for t in texts]
            max_len = max(len(e) for e in encoded)
        
        # Encode all with padding
        encoded = [
            self.encode(t, max_length=max_len, padding=padding)
            for t in texts
        ]
        
        if return_tensors == 'pt':
            return torch.stack([torch.tensor(e, dtype=torch.long) for e in encoded])
        
        return encoded
    
    def get_attention_mask(
        self, 
        token_ids: Union[List[int], torch.Tensor]
    ) -> torch.Tensor:
        """
        Create attention mask (1 for real tokens, 0 for padding).
        
        Args:
            token_ids: Token IDs
            
        Returns:
            Attention mask tensor
        """
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # 1 where not padding, 0 where padding
        mask = (token_ids != self.pad_token_id).long()
        
        return mask
    
    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = True,
        return_tensors: Optional[str] = None,
        return_attention_mask: bool = False
    ) -> dict:
        """
        Tokenize text(s) and return dict matching HuggingFace format.
        
        Args:
            text: Single text or list of texts
            max_length: Maximum sequence length
            padding: Whether to pad
            return_tensors: If 'pt', return tensors
            return_attention_mask: Whether to include attention mask
            
        Returns:
            Dict with 'input_ids' and optionally 'attention_mask'
        """
        if isinstance(text, str):
            input_ids = self.encode(
                text, 
                max_length=max_length, 
                padding=padding,
                return_tensors=return_tensors
            )
            if return_tensors == 'pt' and input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        else:
            input_ids = self.batch_encode(
                text,
                max_length=max_length,
                padding=padding,
                return_tensors=return_tensors
            )
        
        result = {'input_ids': input_ids}
        
        if return_attention_mask:
            if isinstance(input_ids, torch.Tensor):
                result['attention_mask'] = self.get_attention_mask(input_ids)
            else:
                result['attention_mask'] = [
                    self.get_attention_mask(torch.tensor(ids)).tolist()
                    for ids in input_ids
                ]
        
        return result
