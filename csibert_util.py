from torch.nn.utils.rnn import pad_sequence
import torch

import numpy as np

from transformers import BertConfig, BertModel
from torch import nn

from tqdm.notebook import tqdm


def tokenize_csi_matrix(csi_matrix):
    '''
    Tokenize CSI matrix into a 1D sequence
    '''
    # Normalize CSI matrix
    csi_matrix = (csi_matrix - csi_matrix.mean()) / csi_matrix.std()
    # Flatten CSI matrix (time × subcarriers × antennas) to a 1D sequence
    return csi_matrix.reshape(-1)

def pad_sequences(sequences, padding_value=0):
    # Ensure sequences are lists of lists
    if not all(isinstance(seq, list) for seq in sequences):
        raise ValueError("All sequences must be lists")
    
    # Convert sequences to PyTorch tensors
    tensors = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    # Pad to the longest sequence in the batch
    padded_tensors = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    return padded_tensors

def create_attention_mask(padded_sequences):
    # Mask is 1 for non-padding tokens and 0 for padding tokens
    return (padded_sequences != 0).float()

def collate_fn(batch):
    # Batch contains tuples of (inputs, labels)
    inputs, labels = zip(*batch)
    
    # Pad inputs and labels
    padded_inputs = pad_sequences(inputs, padding_value=0)
    padded_labels = pad_sequences(labels, padding_value=0)
    
    # Create attention masks
    attention_mask = create_attention_mask(padded_inputs)
    
    return padded_inputs, padded_labels, attention_mask

class CSIBERT(nn.Module):
    def __init__(self, feature_dim):
        super(CSIBERT, self).__init__()
        self.config = BertConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=4096  # Large enough to accommodate varying lengths
        )
        self.bert = BertModel(self.config)

        # Embedding layers
        # self.time_embedding = nn.Embedding(1024, 768)  # Adjust based on max sequence length
        # self.feature_embedding = nn.Linear(feature_dim, 768)  # Map feature dimension to hidden size

        self.time_embedding = nn.Embedding(1024, 768)
        self.feature_embedding = nn.Linear(feature_dim, 768)


        # Final output layer for regression
        # Update output layer
        self.output_layer = nn.Linear(768, feature_dim)  # Predict all features per token


    def forward(self, inputs, attention_mask=None, output_attentions=False):
        # Input shape: (batch_size, sequence_length, feature_dim)
        batch_size, sequence_length, feature_dim = inputs.shape

        # Generate time embeddings
        time_indices = torch.arange(sequence_length, device=inputs.device).unsqueeze(0)  # Shape: (1, sequence_length)
        time_embeds = self.time_embedding(time_indices).expand(batch_size, -1, -1)  # Shape: (batch_size, sequence_length, hidden_size)

        # Embed feature dimension
        feature_embeds = self.feature_embedding(inputs)  # Shape: (batch_size, sequence_length, hidden_size)

        # Combine embeddings
        combined_embeds = time_embeds + feature_embeds  # Shape: (batch_size, sequence_length, hidden_size)

        # Pass through BERT
        outputs = self.bert(inputs_embeds=combined_embeds,
                            attention_mask=attention_mask,
                            output_attentions=output_attentions)
        
        hidden_states = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)

        attentions = outputs.attentions if output_attentions else None

        # Predict real and imaginary components
        predictions = self.output_layer(hidden_states)  # Shape: (batch_size, sequence_length, 2)

        if output_attentions:
            return predictions, attentions
        
        return predictions

