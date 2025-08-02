import torch
import logging

import numpy as np
import json
import random
import argparse
from datasets import Dataset

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument("--stage", type=str, default="stage1")
    parser.add_argument("--task", type=str, default="vsp-spatial-reasoning", choices=["vsp-spatial-reasoning", "vsp-spatial-planning", "blink-jigsaw", "sat"])

    parser.add_argument("--latent_size", type=int, default=4)
    parser.add_argument("--compress_strategy", type=str, default='average', choices=['average'])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)  # Gradient accumulation steps
    
    parser.add_argument("--epochs", type=int, default=10)   
    
    parser.add_argument("--data_path", type=str, default='PathToJsonlData')    
    parser.add_argument("--log_file", type=str, default='./log.txt')
    
    parser.add_argument("--save_model_path", type=str, default='./checkpoints/model_stage1')
    parser.add_argument("--load_model_path", type=str, default='./checkpoints/model_stage1')
    
    parser.add_argument("--cache_dir", type=str, default='./cache')

    return parser.parse_args()

def seed_everything(seed: int = 42):
    """
    Set seed for reproducibility across random, numpy, torch, and environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_jsonl_dataset(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        data = data[:]
    return Dataset.from_list(data)

def place_input_image(text, image_pad="<|vision_start|><|image_pad|><|vision_end|>", image_placeholder="<image>", sep_token="<|im_start|>assistant") -> str:

    if sep_token is not None:
        assert sep_token in text

        t1, t2 = text.split(sep_token)

        if image_placeholder in t1:
            t1 = t1.replace(image_pad, '')
            t1 = t1.replace(image_placeholder, image_pad)

        return t1 + sep_token + t2
    else:
        return text.replace(image_pad, '').replace(image_placeholder, image_pad)

def place_output_image(text, image_pad="<|vision_start|><|image_pad|><|vision_end|>", latent_placeholder="<output_image>", sep_token="<|im_start|>assistant") -> str:

    if latent_placeholder in text:
        text = text.replace(image_pad+'<think>', '<think>')
        text = text.replace(latent_placeholder, image_pad)

    return text

def remove_user_images(examples):
    new_examples = []
    for example in examples:
        # `example` is a list of turn dicts
        new_example = []
        for turn in example:
            # Create a shallow copy of the turn so we don't modify the original
            new_turn = dict(turn)
            if turn.get("role") == "user":
                # Filter out image-type content
                new_turn["content"] = [
                    item for item in turn.get("content", [])
                    if item.get("type") != "image"
                ]
            # Add the updated turn to this new example
            new_example.append(new_turn)
        new_examples.append(new_example)

    return new_examples

def remove_assistant_images(examples):
    new_examples = []
    for example in examples:
        # `example` is a list of turn dicts
        new_example = []
        for turn in example:
            # Create a shallow copy of the turn so we don't modify the original
            new_turn = dict(turn)
            if turn.get("role") == "assistant":
                # Filter out image-type content
                new_turn["content"] = [
                    item for item in turn.get("content", [])
                    if item.get("type") != "image"
                ]
            # Add the updated turn to this new example
            new_example.append(new_turn)
        new_examples.append(new_example)

    return new_examples

def replace_visual_spectial_tokens(texts):

    update_texts = []
    for i, text in enumerate(texts):
        prev, after = text.split("<|im_start|>assistant")
        update_texts.append(prev + "<|im_start|>assistant" + after.replace("<|vision_start|><|image_pad|><|vision_end|>", "<|latent_start|><|image_pad|><|latent_end|>"))
        
    return update_texts

def replace_subsequent_image_parts_2d(
    input_ids: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    replacement_length: int,
    pad_token: int = 0
) -> torch.Tensor:
    """
    Applies `replace_subsequent_image_parts_1d` to each row of the 2D tensor [batch_size, seq_len],
    then pads all resulting rows to the same max length, returning a single 2D tensor
    of shape [batch_size, new_max_len].
    """
    batch_size, seq_len = input_ids.shape
    
    # Process each row individually, store the result in a list
    replaced_sequences = []
    max_len = 0
    
    for i in range(batch_size):
        seq_1d = input_ids[i]
        new_seq_1d = replace_subsequent_image_parts_1d(
            seq_1d,
            start_token=start_token,
            end_token=end_token,
            replacement_token=replacement_token,
            replacement_length=replacement_length
        )
        replaced_sequences.append(new_seq_1d)
        max_len = max(max_len, new_seq_1d.size(0))
    
    # Now pad all replaced sequences to 'max_len'
    # We'll create a new tensor on the same device, same dtype
    new_input_ids = input_ids.new_full((batch_size, max_len), fill_value=pad_token)
    
    for i, seq_1d in enumerate(replaced_sequences):
        length = seq_1d.size(0)
        new_input_ids[i, :length] = seq_1d
    
    return new_input_ids

def replace_subsequent_image_parts_1d(
    seq: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    replacement_length: int
) -> torch.Tensor:
    """
    Process a single 1D sequence, replacing everything from start_token..end_token
    with replacement_length copies of replacement_token. 
    """
    # Find positions of start and end tokens
    start_positions = (seq == start_token).nonzero().squeeze(-1)
    end_positions   = (seq == end_token).nonzero().squeeze(-1)

    new_seq_pieces = []
    prev_end = 0
    
    for (s_pos, e_pos) in zip(start_positions, end_positions):
        # Add everything before this image part as-is, BUT include start_token itself
        new_seq_pieces.append(seq[prev_end : s_pos + 1])
        
        # Replace the entire chunk [s_pos+1 .. e_pos) with N copies of replacement_token
        replacement_span = torch.tensor(
            [replacement_token] * replacement_length, 
            dtype=seq.dtype, 
            device=seq.device
        )
        new_seq_pieces.append(replacement_span)

        # Move past the end_token
        prev_end = e_pos
    
    # Add whatever remains after the last image part
    if prev_end < len(seq):
        new_seq_pieces.append(seq[prev_end:])
    
    # Concatenate into a single 1D tensor
    new_seq = torch.cat(new_seq_pieces, dim=0)
    return new_seq

def process_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    replacement_length: int,
    pad_token: int = 0
):

    batch_size, seq_len = input_ids.shape

    # We'll accumulate the processed (variable-length) sequences here.
    processed_sequences = []

    for b in range(batch_size):
        # Extract the unpadded tokens using the attention mask (real tokens only)
        real_tokens = input_ids[b][attention_mask[b] == 1]

        # Perform image-part replacement on this 1D sequence
        updated_seq = replace_subsequent_image_parts_1d(
            real_tokens,
            start_token=start_token,
            end_token=end_token,
            replacement_token=replacement_token,
            replacement_length=replacement_length
        )

        processed_sequences.append(updated_seq)

    # Now we have a list of 1D tensors of different lengths.
    # We'll re-pad them so they can be stacked into a [batch_size, new_seq_len] tensor.
    new_max_len = max(seq.size(0) for seq in processed_sequences)

    # Create new tensors for input_ids and attention_mask.
    # We'll fill input_ids with the specified pad_token.
    new_input_ids = input_ids.new_full((batch_size, new_max_len), fill_value=int(pad_token))
    # For attention_mask, padded positions are 0 by definition
    new_attention_mask = input_ids.new_zeros((batch_size, new_max_len))

    # Copy each processed sequence back into these padded tensors.
    for b in range(batch_size):
        seq_len_b = processed_sequences[b].size(0)
        new_input_ids[b, :seq_len_b] = processed_sequences[b]
        new_attention_mask[b, :seq_len_b] = 1

    return new_input_ids, new_attention_mask

def find_subsequence(row: torch.Tensor, pattern: torch.Tensor) -> int:

    seq_len = row.size(0)
    pat_len = pattern.size(0)
    
    # Naive scan over all possible start positions
    for start_idx in range(seq_len - pat_len + 1):
        # Compare row[start_idx : start_idx + pat_len] to pattern
        if torch.all(row[start_idx : start_idx + pat_len] == pattern):
            return start_idx
    return -1

def generate_labels_after_multi_token_start(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    pad_token_idx: int = 0,
    img_token_idx: int = 151655,
) -> torch.Tensor:
    """
    For each row in `input_ids`, find the *first* occurrence of `start_sequence`
    (a 1D tensor of multiple token IDs). Mask all tokens up to and including
    that entire sub-sequence (set them to -100), and also mask any padding tokens
    anywhere in the row. The remainder (tokens *after* the sub-sequence) are kept.

    Args:
      input_ids: 2D tensor [batch_size, seq_len].
      start_sequence: 1D tensor of shape [k], the multi-token "start" pattern.
      pad_token_id: which ID is used as padding (default=0).
    
    Returns:
      labels: a new 2D tensor [batch_size, seq_len], where tokens before (and
              including) the sub-sequence are -100, as well as any pad tokens,
              and tokens after the sub-sequence are kept as in `input_ids`.
    """
    batch_size, seq_len = input_ids.shape
    
    # Clone so we can modify in-place
    labels = input_ids.clone()
    
    for b in range(batch_size):
        row = labels[b]
        # Find first occurrence of the entire sub-sequence
        start_idx = find_subsequence(row, start_sequence)
        
        if start_idx == -1:
            # Sub-sequence not found -> mask everything
            row[:] = -100
        else:
            # The sub-sequence length
            sub_len = start_sequence.size(0)
            end_of_subseq = start_idx + sub_len  # the position *after* the sub-sequence
            
            # Mask everything up to (and including) the sub-sequence
            row[:end_of_subseq] = -100
        
        # Mask pad tokens
        row[row == pad_token_idx] = -100
        # Mask image tokens
        row[row == img_token_idx] = -100
    
    return labels

def generate_labels_after_latent_tokens(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    pad_token_idx: int = 0,
) -> torch.Tensor:
    """
    For each row in `input_ids`, find the *first* occurrence of `start_sequence`
    (a 1D tensor of multiple token IDs). Mask all tokens up to and including
    that entire sub-sequence (set them to -100), and also mask any padding tokens
    anywhere in the row. The remainder (tokens *after* the sub-sequence) are kept.

    Args:
      input_ids: 2D tensor [batch_size, seq_len].
      start_sequence: 1D tensor of shape [k], the multi-token "start" pattern.
      pad_token_id: which ID is used as padding (default=0).
    
    Returns:
      labels: a new 2D tensor [batch_size, seq_len], where tokens before (and
              including) the sub-sequence are -100, as well as any pad tokens,
              and tokens after the sub-sequence are kept as in `input_ids`.
    """
    batch_size, seq_len = input_ids.shape
    
    # Clone so we can modify in-place
    labels = input_ids.clone()
    
    for b in range(batch_size):
        row = labels[b]
        # Find first occurrence of the entire sub-sequence
        start_idx = find_subsequence(row, start_sequence)
        
        if start_idx == -1:
            # Sub-sequence not found -> mask everything
            row[:] = -100
        else:
            row[:start_idx] = -100
        
        # Mask pad tokens
        row[row == pad_token_idx] = -100
    
    return labels

def mask_image_output_tokens(
    input_ids: torch.Tensor,
    image_start_token: int,
    image_token: int
) -> torch.Tensor:
    """
    Creates a mask of the same shape as `input_ids`, with 1's wherever we want to
    'mask out' <image_token> after the first <image_start_token> has appeared,
    and 0's everywhere else.

    Args:
      input_ids: shape [batch_size, seq_len]
      image_start_token: the token ID that marks the start of an image chunk
      image_token: the token ID for image tokens

    Returns:
      A mask (torch.Tensor of the same shape) containing 0/1:
        - 1 = this position should be masked
        - 0 = this position is kept
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids)

    for i in range(batch_size):
        seq = input_ids[i]
        # Find first occurrence of image_start_token
        first_start_pos = -1
        for j in range(seq_len):
            if seq[j] == image_start_token:
                first_start_pos = j
                break
        
        if first_start_pos == -1:
            continue
        
        # For every position after the first <image_start_token>,
        # if the token is <image_token>, set mask = 1
        for k in range(first_start_pos + 1, seq_len):
            if seq[k] == image_token:
                mask[i, k] = 1

    return mask


if __name__=="__main__":
    
    pass