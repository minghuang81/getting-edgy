# Run a background UDP client to continously receive realtime radar data;
# In the main thread, continously train a BERT Encoder and evaluate the MSE loss. 
# If the MSE loss shows a spike, it would mean that the input is a stranger
# to the model. After the model frequenly 'sees' an activity, the associated
# MSE loss will be low.

# Summary of The Setup
#     Input: Sequences of embeddings (shape [batch_size, seq_len, hidden_size])
#     Goal: Predict masked embeddings (shape [batch_size, seq_len, hidden_size])
#     Loss: Mean squared error (MSE) on masked positions only
#     Output: Use outputs.last_hidden_state instead of vocab logits
#     Model: Use RobertaModel, not RobertaForMaskedLM   

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from transformers import AdamW
from torch.optim import AdamW 
import math
import datetime
import os

# Use GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

sav_state = "../pytorch_saved_model/weight_model.pt"
logfile = "../pytorch_saved_model/log.txt"

FRAME_LEN = 256     # number of chirps to include in one sequence
HIDDEN_SIZE = 32
FF_DIM = 2*HIDDEN_SIZE
NUM_LAYERS = 1
NUM_HEADS = 2
EPOCHS = 30

mti_a = 0.01        # MTI weight for heartbeat, Fc 0.8Hz, >48 BPM
low_a = 0.05        # lowpass Fc 2.7Hz, <180 BPM
mti_H = None
low_H = None


# Rebuild Transformer Encoder model
# __________________________
# Fully custom TransformerEncoderLayer and CustomTransformerEncoder implemented
# using basic PyTorch building blocks (nn.Linear, nn.LayerNorm, nn.MultiheadAttention, etc.),
# mimicking the behavior of PyTorch's nn.TransformerEncoder

# The positional encoding is registered as a buffer (not trainable).
# You can still set max_seq_len in the constructor if needed.
# This style mimics the original Transformer from Vaswani et al., not BERT/RoBERTa (which use learned embeddings).
def sinusoidal_positional_encoding(seq_len, hidden_size, device):
    pe = torch.zeros(seq_len, hidden_size, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(
        0, hidden_size, 2, device=device).float() * (-math.log(10000.0) / hidden_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # shape: [seq_len, hidden_size]

# MultiheadAttention   : Learn contextual relations between tokens
# LayerNorm + residual : Stabilizes training and supports deep stacking
# Feedforward (MLP)    : Adds non-linearity and depth to each token's features
# Dropout              : Regularization
# key_padding_mask     : Tells attention to ignore padding tokens
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Self-attention sublayer
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feedforward sublayer
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x

class CustomTransformerEncoder(nn.Module):
    def __init__(self, seq_len, hidden_size=64, num_layers=2, num_heads=2, 
                  ff_dim=256, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size)
        )
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(hidden_size, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.register_buffer("positional_encoding", self._init_positional_encoding(device))

    def _init_positional_encoding(self, device):
        return sinusoidal_positional_encoding(self.seq_len, self.hidden_size, device=device)

    def forward(self, inputs_embeds, attention_mask=None):
        # Convert attention mask (1 = keep, 0 = pad) to key_padding_mask (True = pad, False = keep)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        pe = self.positional_encoding[:inputs_embeds.size(1), :]\
                    .unsqueeze(0).to(inputs_embeds.device)  # shape: [1, S, H]
        x = inputs_embeds + pe
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        return self.norm(x)  # final layer norm (optional)
    

model = CustomTransformerEncoder(seq_len=FRAME_LEN, hidden_size=HIDDEN_SIZE, 
                                  num_layers=NUM_LAYERS, num_heads=NUM_HEADS, 
                                  ff_dim=FF_DIM).to(device)

    
# FFT of each 128-sample chirp and filter along slowtime axis
# __________________________________________________________

def process_chirp(chirp, hidden_size) : # chirp.shape : (hidden_size,)
    global mti_H, low_H
    if mti_H is None:
        mti_H = np.zeros(hidden_size, dtype=np.complex64)
        low_H = np.zeros(hidden_size, dtype=np.complex64)
    
    # Step 1: Perform FFT along the range dimension (128 samples)
    chirp = chirp / 4096 # radar sample is 12-bit integer
    range_fft = np.fft.fft(chirp) 
    range_fft = range_fft[:hidden_size] # truncate
   
    # Step 2: highpass for heartbeat
    mti_H = mti_a * range_fft + (1-mti_a) * mti_H
    range_fft -= mti_H
    
    # Step 3: lowpass for heartbeat and breath
    low_H = low_a * range_fft + (1-low_a) * low_H
    range_fft = low_H

    # Step 4-: range bins partial distance compensation
    eq = np.array([np.sqrt(n) for n in range(1, hidden_size+1)])
    range_fft = eq * range_fft
    
    return range_fft.real    # array of floats of len hidden_size

# Custom Dataset (Loads Float Embeddings from File)
# __________________________________________

class EmbeddingDataset(Dataset):
    def __init__(self, data_buffer, seq_len=10, hidden_size=128):
        self.data_buffer = data_buffer
        self.seq_len = seq_len
        self.hidden_size = hidden_size

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        y = np.array(data_buffer).astype(np.float32)
        example = torch.tensor(y).view(-1, self.seq_len, self.hidden_size)
        if example.shape[0] != 1:
            print(f"Bad sample shape {example.shape}")
            stop_event.set()
        return example[0]  # shape(seq_len, hidden_size)


# Data Collator for Embedding Reconstruction
# __________________________________________

class EmbeddingReconstructionCollator:
    def __init__(self, mlm_probability=0.15):
        self.mlm_probability = mlm_probability

    def __call__(self, batch):
        # batch: list of [seq_len, hidden_size]
        batch_embeddings = torch.stack(batch)  # (B, S, H)
        inputs = batch_embeddings.clone()
        mask = torch.rand(inputs.shape[:2]) < self.mlm_probability  # (B, S)

        # Replace masked positions with zero vectors
        inputs[mask] = 0.0

        return {
            "inputs_embeds": inputs,               # Masked inputs
            "target_embeddings": batch_embeddings, # Ground-truth to reconstruct
            "mask": mask,                          # Boolean mask: where to compute loss
            "attention_mask": torch.ones(inputs.shape[:2])
        }


# UDP client to the PSOC6 server
# ______________________________
import socket
import threading
from collections import deque

# IP details for the UDP server
DEFAULT_IP   = '192.168.1.43'   # IP address of the UDP server
discovered_IP = ""              # Discovered IP address of the UDP server
DEFAULT_PORT = 57345            # Port of the UDP server
BROADCAST_IP = '192.168.1.255'  # Broadcast address
START_FRAME="A"
END_FRAME="Z"
DISCOVERY_MSG = b'DISCOVERY_REQUEST'
BUFFER_SIZE = 1500
Nc = 128            # nb of samples per chirp

def udp_acquisition(hidden_size=HIDDEN_SIZE):
    discovered_IP = ""
    while not stop_event.is_set():
        discovered_IP = who_is_server(DEFAULT_PORT)
        if discovered_IP != "":
            #start udp client
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(10)    # wait for the next frame, may wait unitl 6s
            s.sendto(bytes(START_FRAME, "utf-8"), (discovered_IP, DEFAULT_PORT))
            while not stop_event.is_set():
                try:
                    data = s.recv(BUFFER_SIZE)
                    frame = np.frombuffer(data, dtype=np.uint16)
                    frame = frame.astype(np.float32)
                    chirps = frame.reshape((-1, Nc))
                    for i in range(chirps.shape[0]):
                        range_bin = process_chirp(chirps[i], hidden_size)
                        data_buffer.append(range_bin)
                    if len(data_buffer) == FRAME_LEN: 
                        start_event.set()
                except socket.timeout:
                    print("Frame fragment not received. retry!")
                    break # continue to IP discovery
            s.close()
    udp_exit(discovered_IP, DEFAULT_PORT )

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        s.connect((DEFAULT_IP, 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

            
def who_is_server(server_port):
    my_IP = get_ip_address()
    BROADCAST_IP = '.'.join(my_IP.split('.')[:-1]+['255'])
    print(f"My IP: {my_IP}, using Broadcast IP Address: {BROADCAST_IP}, Port: {server_port}")
    print("================================================================================")

    # Create UDP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  # Enable broadcast
    
    # Optionally set a timeout for responses
    s.settimeout(3.0)
    
    # Send the broadcast message
    s.sendto(DISCOVERY_MSG, (BROADCAST_IP, server_port))
    print(f"Sent discovery message to {BROADCAST_IP}:{server_port}")
    
    # Listen for response
    try:
        data, addr = s.recvfrom(BUFFER_SIZE)  # Buffer size
        addr_from = addr[0]
        addr_srv = data.decode('utf-8')
        
        if addr_from == addr_srv:   # yes, it's the UDP server
            s.close()
            print(f"Received response from {addr}: {addr_srv}")
            return addr_srv
        else:
            print(f"Discard response from {addr_from}: {addr_srv}")
    except socket.timeout:
        print("No response received.")
    s.close()
    return ""

def udp_exit(server_ip, server_port):
    print("The script is exiting... Goodbye!")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(bytes(END_FRAME, "utf-8"), (server_ip, server_port))
    s.close()
    
# Training Loop with MSE on Masked Embeddings
# __________________________________________

def saveModel():
    total_params = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), sav_state)
    logMsg(f"Model is saved in {sav_state}. av loss {av_loss:.4f}, total params {total_params}")
        
def loadModel():
    dir_path = os.path.dirname(sav_state)
    fprefix = os.path.basename(sav_state)
    pt_files = [f for f in os.listdir(dir_path) if f.startswith(fprefix)]
    pt_files.sort()
    if pt_files:
        fname = dir_path + '/' + pt_files[-1]
        logMsg(f"Model states loaded from {fname}")
        model.load_state_dict(torch.load(fname, map_location=device))
    else:
        logMsg(f"Model states: none exists in {dir_path}")

def logMsg(msg):
    print(msg)
    with open(logfile, 'a') as f:
        print(msg, file=f)
    
if __name__ == '__main__':
    # load previously partially trained model and continue from there
    loadModel()
    
    # Shared data buffer: udp_acquisition -> data_buffer -> dataset
    data_buffer = deque(maxlen=FRAME_LEN)
    thread = threading.Thread(target=udp_acquisition, daemon=True)
    stop_event = threading.Event()
    start_event = threading.Event()
    thread.start()
    
    # Start Detection
    
    dataset = EmbeddingDataset(data_buffer, seq_len=FRAME_LEN, hidden_size=HIDDEN_SIZE)
    collator = EmbeddingReconstructionCollator(mlm_probability=0.15)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collator)
    
    #optimizer = AdamW(model.parameters(), lr=1e-4)
    optimizer = AdamW(model.parameters(), lr=1e-6)
    model.train()   # paired with model.eval() during validation/testing
    
    av_loss = 1
    av_av_loss = av_loss
    ntrain = 0
    try:
        # for i in range(100):
        while True:
            while not start_event.is_set():
                start_event.wait(0.1)
            start_event.clear()
            batch = collator([dataset[0]])
            inputs_embeds = batch["inputs_embeds"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_embeddings = batch["target_embeddings"].to(device)
            mask = batch["mask"].to(device)
            
            preds = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        
            # Compute loss only on masked positions
            loss = F.mse_loss(preds[mask], target_embeddings[mask])
            av_loss = 0.008 * loss.item() + (1-0.008) * av_loss
            av_av_loss = 0.0005 * loss.item() + (1-0.0005) * av_av_loss
            if av_loss > av_av_loss*3 :
                t = f"{datetime.datetime.now()}".replace("-","")[:21]
                logMsg(f"{t} : Activity change detected ! av_loss {av_loss} >> av_av_loss {av_av_loss}")
                av_av_loss = av_loss # reset the threshold to avoid redundant detection
                
            if loss.item() > av_loss:  # train smart
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                ntrain += 1
            
            # bookkeeping
            if (ntrain & 0xFF) == 0:
                t = f"{datetime.datetime.now()}".replace("-","")[:21]
                logMsg(f"{t} #train: {ntrain}, Loss: {loss.item():.4f}, av_loss: {av_loss:.04f}, av_av_loss: {av_av_loss:.04f}")
                if (ntrain & 0xFFFF) == 0:
                    saveModel()
                ntrain += 1 # a hack to not repeat the same print
    except KeyboardInterrupt:
        print("Ctrl-C detected! Stopping threads...")
        stop_event.set()
    finally:
        stop_event.set()
        thread.join()      # Wait for thread to finish
        print("Main process done.")




