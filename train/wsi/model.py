import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")

# -------------------------------
# Bulk Data Loading and Matching
# -------------------------------
def load_bulk_data(bulk_csv_path):
    bulk_df = pd.read_csv(bulk_csv_path)
    bulk_df["gene_name"] = bulk_df['Unnamed: 0'].str[:15]
    bulk_df = bulk_df.drop(columns=['Unnamed: 0'])
    bulk_df = bulk_df.set_index('gene_name')
    
    original_ids = list(bulk_df.columns)
    patient_ids = [pid[:19] for pid in original_ids]
    patient_id_series = pd.Series(patient_ids)
    
    duplicate_ids = patient_id_series[patient_id_series.duplicated()].unique()
    print("Duplicate patient IDs:", duplicate_ids)
    print("Patient ID counts:\n", patient_id_series.value_counts())
    
    valid_patient_ids = patient_id_series[~patient_id_series.isin(duplicate_ids)].unique()
    valid_original_ids = [oid for oid in original_ids if oid[:19] in valid_patient_ids]
    bulk_df = bulk_df[valid_original_ids]
    
    return bulk_df, valid_patient_ids

def extract_patient_id(file_path):
    return os.path.basename(file_path)[0:19]

def find_parquet_files(folder_path, valid_patient_ids):
    parquet_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".parquet"):
                full_path = os.path.join(root, file)
                pid = extract_patient_id(full_path)
                if pid in valid_patient_ids:
                    parquet_files.append(full_path)
    return parquet_files

# -------------------------------
# Dataset Definition
# -------------------------------
class WsiDataset(Dataset):
    def __init__(self, wsi_dir, rnaseq_csv, transform=None, max_samples=None, max_cells=60000):
        super().__init__()
        self.transform = transform
        self.max_cells = max_cells
        print(f"Received max_samples: {max_samples}")
        
        self.rnaseq_df, self.valid_patient_ids = load_bulk_data(rnaseq_csv)
        all_files = find_parquet_files(wsi_dir, self.valid_patient_ids)
        print(f"Total parquet files found: {len(all_files)}")
        
        wsi_dict = {}
        for f in all_files:
            pid = extract_patient_id(f)
            if pid in wsi_dict:
                wsi_dict[pid].append(f)
            else:
                wsi_dict[pid] = [f]
        
        self.data_list = []
        for pid in self.valid_patient_ids:
            if pid in wsi_dict:
                bulk_col = [col for col in self.rnaseq_df.columns if col[:19] == pid]
                if len(bulk_col) == 1:
                    bulk_expr = self.rnaseq_df[bulk_col[0]].values.astype(np.float32)
                    for wsi_file in wsi_dict[pid]:
                        self.data_list.append((pid, wsi_file, bulk_expr))
                else:
                    print(f"Warning: Patient {pid} has {len(bulk_col)} bulk columns, skipping.")
        
        print(f"Initial data_list length: {len(self.data_list)}")
        if max_samples is not None:
            print(f"Applying max_samples limit: {max_samples}")
            self.data_list = self.data_list[:max_samples]
        print(f"Total matched WSI samples: {len(self.data_list)}")
        if len(self.data_list) == 0:
            raise ValueError("No valid WSI samples matched. Check ID extraction and data alignment.")
        
        self.gene_list = self.rnaseq_df.index.tolist()
        self.num_genes = len(self.gene_list)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        pid, wsi_file, bulk_expr = self.data_list[idx]
        feature_columns = [f'feature_{i}' for i in range(0, 128)] + ['cluster_label']
        cell_df = pd.read_parquet(wsi_file, columns=feature_columns)
        
        if len(cell_df) > self.max_cells:
            cell_df = cell_df.sample(n=self.max_cells, random_state=42)
        
        cell_features = cell_df[[f'feature_{i}' for i in range(0, 128)]].values.astype(np.float32)
        cluster_labels = cell_df['cluster_label'].values.astype(np.int64)
        
        cell_features = torch.tensor(cell_features)
        cluster_labels = torch.tensor(cluster_labels)
        
        if self.transform:
            cell_features = self.transform(cell_features)
        
        sample = {
            "patient_id": pid,
            "cell_features": cell_features,
            "cluster_labels": cluster_labels,
            "bulk_expr": torch.tensor(bulk_expr)
        }
        return sample

def collate_fn(batch):
    patient_ids = [b["patient_id"] for b in batch]
    bulk_exprs = torch.stack([b["bulk_expr"] for b in batch], dim=0)
    cell_features_list = [b["cell_features"] for b in batch]
    cluster_labels_list = [b["cluster_labels"] for b in batch]
    
    lengths = [cf.shape[0] for cf in cell_features_list]
    padded_cells = pad_sequence(cell_features_list, batch_first=True)
    padded_labels = pad_sequence(cluster_labels_list, batch_first=True, padding_value=-1)
    
    mask = torch.zeros(padded_cells.shape[:2], dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    
    return {
        "patient_ids": patient_ids,
        "cell_features": padded_cells,
        "cell_mask": mask,
        "cluster_labels": padded_labels,
        "bulk_expr": bulk_exprs
    }

# -------------------------------
# Transformer Model with Gradient Checkpointing
# -------------------------------
class TransformerPredictor(nn.Module):
    def __init__(self, input_dim=128, embed_dim=256, num_genes=1000, num_layers=3, nhead=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, num_genes)
        self.relu = nn.ReLU()
        self.embed_dim = embed_dim

    def generate_pos_embedding(self, N):
        position = torch.arange(N, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.embed_dim))
        pos_embedding = torch.zeros(N, self.embed_dim)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        return pos_embedding.unsqueeze(0)

    def encoder_forward(self, x, src_key_padding_mask):
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def forward(self, cell_features, cell_mask):
        B, N, _ = cell_features.shape
        x = self.input_proj(cell_features)
        pos = self.generate_pos_embedding(N).to(x.device)
        x = x + pos
        src_key_padding_mask = ~cell_mask
        # Use checkpointing on the entire encoder
        x = checkpoint(self.encoder_forward, x, src_key_padding_mask, use_reentrant=False)
        cell_pred = self.fc_out(x)
        cell_pred = self.relu(cell_pred)
        return cell_pred

# -------------------------------
# Training Function with Mixed Precision and GPU Clustering Loss
# -------------------------------
def train_model(model, train_loader, test_loader, optimizer, num_epochs=10, device="cuda", cells_per_block=5000, cluster_loss_weight=1.0):
    model.to(device)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    best_loss = float('inf')
    
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            cell_features = batch["cell_features"].to(device)
            cell_mask = batch["cell_mask"].to(device)
            cluster_labels = batch["cluster_labels"].to(device)
            bulk_expr = batch["bulk_expr"].to(device)
            
            optimizer.zero_grad()
            B, N, _ = cell_features.shape
            assert B == 1, "Currently only supports batch_size=1"
            agg_pred = torch.zeros(1, bulk_expr.shape[1], device=device)
            cluster_loss = 0.0
            
            for start in range(0, N, cells_per_block):
                end = min(start + cells_per_block, N)
                block_features = cell_features[:, start:end, :]
                block_mask = cell_mask[:, start:end]
                block_labels = cluster_labels[:, start:end]
                
                with autocast():
                    cell_pred = model(block_features, block_mask)
                    # Squeeze batch dimension since B=1
                    cell_pred = cell_pred[0]
                    block_mask = block_mask[0]
                    block_labels = block_labels[0]
                    
                    # Compute clustering loss on GPU for this block
                    valid_preds = cell_pred[block_mask]
                    valid_labels = block_labels[block_mask]
                    unique_labels = torch.unique(valid_labels)
                    for label in unique_labels:
                        if label == -1:
                            continue
                        cluster_preds = valid_preds[valid_labels == label]
                        if cluster_preds.size(0) > 1:
                            centroid = cluster_preds.mean(dim=0)
                            distances = torch.norm(cluster_preds - centroid, dim=1)
                            cluster_loss += distances.mean()
                    
                    # Aggregate for prediction loss
                    cell_pred_masked = cell_pred * block_mask.unsqueeze(-1).float()
                    agg_pred += cell_pred_masked.sum(dim=0, keepdim=True)
            
            with autocast():
                # Normalize agg_pred
                sum_agg_pred = agg_pred.sum()
                normalized_agg_pred = agg_pred / sum_agg_pred
                result = normalized_agg_pred * 1000000
                pred_loss = criterion(result, bulk_expr)
                total_loss = pred_loss + cluster_loss_weight * cluster_loss
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += total_loss.item() * B
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                cell_features = batch["cell_features"].to(device)
                cell_mask = batch["cell_mask"].to(device)
                bulk_expr = batch["bulk_expr"].to(device)
                B, N, _ = cell_features.shape
                assert B == 1, "Currently only supports batch_size=1"
                agg_pred = torch.zeros(1, bulk_expr.shape[1], device=device)
                
                for start in range(0, N, cells_per_block):
                    end = min(start + cells_per_block, N)
                    block_features = cell_features[:, start:end, :]
                    block_mask = cell_mask[:, start:end]
                    with autocast():
                        cell_pred = model(block_features, block_mask)
                        cell_pred = cell_pred[0]
                        block_mask = block_mask[0]
                        cell_pred_masked = cell_pred * block_mask.unsqueeze(-1).float()
                        agg_pred += cell_pred_masked.sum(dim=0, keepdim=True)
                
                with autocast():
                    sum_agg_pred = agg_pred.sum()
                    normalized_agg_pred = agg_pred / sum_agg_pred
                    result = normalized_agg_pred * 1000000
                    loss = criterion(result, bulk_expr)
                test_loss += loss.item() * B
        
        test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "best_transformer_model.pt")
    
    print(f"Training complete. Best Test Loss: {best_loss:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()

def inference(model, cell_features, cell_mask, device="cuda", cells_per_block=500):
    model.to(device)
    model.eval()
    with torch.no_grad():
        if cell_features.dim() == 2:
            cell_features = cell_features.unsqueeze(0)
            cell_mask = cell_mask.unsqueeze(0)
        
        B, N, _ = cell_features.shape
        cell_pred_list = []
        
        for start in range(0, N, cells_per_block):
            end = min(start + cells_per_block, N)
            block_features = cell_features[:, start:end, :].to(device)
            block_mask = cell_mask[:, start:end].to(device)
            with autocast():
                block_pred = model(block_features, block_mask)
            cell_pred_list.append(block_pred.cpu())
        
        cell_pred = torch.cat(cell_pred_list, dim=1)
    
    return cell_pred.squeeze(0)

# -------------------------------
# Main Function
# -------------------------------
def main():
    wsi_dir = "hover_net-master/extracted_features"
    rnaseq_csv = "basic_model/tpm-TCGA-COAD.csv"
    batch_size = 1
    num_epochs = 100
    learning_rate = 1e-5
    max_samples = 300
    cluster_loss_weight = 0.1
    
    dataset = WsiDataset(wsi_dir, rnaseq_csv, max_samples=max_samples, max_cells=60000)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    num_genes = dataset.num_genes
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TransformerPredictor(input_dim=128, embed_dim=256, num_genes=num_genes, num_layers=3, nhead=8, dropout=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_model(model, train_loader, test_loader, optimizer, num_epochs=num_epochs, device=device, cluster_loss_weight=cluster_loss_weight)
    
    sample = dataset[0]
    cell_feats = sample["cell_features"]
    cell_mask = torch.ones(cell_feats.shape[0], dtype=torch.bool)
    model.load_state_dict(torch.load("best_transformer_model.pt", map_location=device))
    pred = inference(model, cell_feats, cell_mask, device=device, cells_per_block=500)
    print("Predicted gene expression shape for a single WSI:", pred.shape)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
# PYTHONWARNINGS=ignore python basic_model/model_with_cluster.py