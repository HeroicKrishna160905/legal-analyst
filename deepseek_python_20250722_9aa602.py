import streamlit as st
import torch
import numpy as np
import pandas as pd
import re
import os
import gc
import ast
import json
import nltk
from transformers import (
    AutoTokenizer, BertForMaskedLM, DistilBertForMaskedLM,
    DistilBertForSequenceClassification, DistilBertForTokenClassification
)
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from kaggle_secrets import UserSecretsClient

# Set up device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nltk.download('punkt', quiet=True)

# Create directories
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Streamlit app configuration
st.set_page_config(
    page_title="Legal Outcome Predictor",
    page_icon="⚖️",
    layout="wide"
)

# Define all classes and functions
class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class RationaleDataset(Dataset):
    def __init__(self, texts, rationale_masks, tokenizer, max_len=512):
        self.texts = texts
        self.rationale_masks = rationale_masks
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        # Get rationale mask for this sample
        mask = self.rationale_masks[idx]
        # Pad or truncate mask to max length
        mask = mask[:self.max_len] + [0] * (self.max_len - len(mask))
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(mask, dtype=torch.long)
        }

class InferenceSystem:
    def __init__(self):
        # Configuration
        self.CLASSIFIER_PATH = "models/classification_model"
        self.RATIONALE_PATH = "models/rationale_model"
        self.DISTILLED_PATH = "models/distilled_model"
        
        # Load components
        self.tokenizer = self._load_tokenizer()
        self.classifier = self._load_classifier()
        self.rationale_extractor = self._load_rationale_extractor()
    
    def _load_tokenizer(self):
        # Try rationale path first, then fallback to classifier
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.RATIONALE_PATH)
            return tokenizer
        except:
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.CLASSIFIER_PATH)
                return tokenizer
            except:
                tokenizer = AutoTokenizer.from_pretrained(self.DISTILLED_PATH)
                return tokenizer
    
    def _load_classifier(self):
        try:
            return DistilBertForSequenceClassification.from_pretrained(
                self.CLASSIFIER_PATH
            ).to(DEVICE).eval()
        except Exception as e:
            print(f"Error loading classifier: {e}")
            return None
    
    def _load_rationale_extractor(self):
        try:
            model = DistilBertForTokenClassification.from_pretrained(
                self.RATIONALE_PATH
            ).to(DEVICE).eval()
            return model
        except Exception as e:
            print(f"Could not load rationale model: {e}")
            print("Using classification model with attention fallback")
            return None
    
    def predict(self, text):
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            padding="max_length", 
            truncation=True
        ).to(DEVICE)
        
        with torch.no_grad():
            # Predict outcome
            if self.classifier is None:
                raise RuntimeError("Classifier model failed to load")
                
            clf_output = self.classifier(**inputs)
            label = clf_output.logits.argmax(-1).item()
            outcome = "ALLOWED" if label == 1 else "DISMISSED"
            
            # Extract rationale
            if self.rationale_extractor:
                token_output = self.rationale_extractor(**inputs)
                mask = token_output.logits.argmax(-1).squeeze().cpu().numpy()
            else:
                # Fallback: use attention weights from classifier
                outputs = self.classifier(**inputs, output_attentions=True)
                attentions = torch.stack(outputs.attentions).mean(0).mean(1)[0, 0].cpu().numpy()
                mask = (attentions > attentions.mean()).astype(int)
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().cpu().numpy())
            attention_mask = inputs["attention_mask"].squeeze().cpu().numpy()
            
            # Filter out special tokens and create highlighted text
            highlighted_tokens = []
            for i, (tok, m) in enumerate(zip(tokens, mask)):
                # Skip special tokens and padding
                if tok in [self.tokenizer.cls_token, 
                          self.tokenizer.sep_token, 
                          self.tokenizer.pad_token] or attention_mask[i] == 0:
                    continue
                    
                # Clean up token representation
                if tok.startswith("##"):
                    tok = tok[2:]
                    if highlighted_tokens:
                        highlighted_tokens[-1] += tok
                        continue
                
                if m == 1:
                    highlighted_tokens.append(f"[{tok}]")
                else:
                    highlighted_tokens.append(tok)
            
            # Convert to readable text
            rationale_text = " ".join(highlighted_tokens)
            # Clean up spacing around punctuation
            rationale_text = re.sub(r'\s+([.,;:!?])', r'\1', rationale_text)
            rationale_text = re.sub(r'\[\s+', '[', rationale_text)
            rationale_text = re.sub(r'\s+\]', ']', rationale_text)
        
        return outcome, rationale_text

@st.cache_resource(show_spinner=False)
def preprocess_data():
    # Securely load Hugging Face token
    user_secrets = UserSecretsClient()
    HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
    
    if not HF_TOKEN:
        st.error("Hugging Face token not found in Kaggle secrets. Please add your HF_TOKEN to Kaggle secrets.")
        st.stop()
    
    # Authenticate with Hugging Face
    login(token=HF_TOKEN)
    
    # Load CJPE dataset
    with st.spinner("Loading CJPE dataset..."):
        dataset = load_dataset("Exploration-Lab/IL-TUR", "cjpe")
    
    # Convert to DataFrames
    df_train = pd.DataFrame(dataset["single_train"])
    df_dev = pd.DataFrame(dataset["single_dev"])
    df_test = pd.DataFrame(dataset["test"])
    df_expert = pd.DataFrame(dataset["expert"]) if "expert" in dataset else None
    
    # Create raw_text column for all splits
    df_train["raw_text"] = df_train["text"]
    df_dev["raw_text"] = df_dev["text"]
    df_test["raw_text"] = df_test["text"]
    
    # Add raw_text for expert split if it exists
    if df_expert is not None:
        df_expert["raw_text"] = df_expert["text"]
    
    # Tokenization
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    MAX_LEN = 512
    
    def tokenize_batch(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
            return_offsets_mapping=True
        )
    
    with st.spinner("Tokenizing data..."):
        train_tokens = tokenize_batch(df_train["raw_text"].tolist())
        dev_tokens = tokenize_batch(df_dev["raw_text"].tolist())
        test_tokens = tokenize_batch(df_test["raw_text"].tolist())
    
    # Initialize rationale masks with zeros
    df_train["rationale_mask"] = [[0]*MAX_LEN for _ in range(len(df_train))]
    df_dev["rationale_mask"] = [[0]*MAX_LEN for _ in range(len(df_dev))]
    
    # Process expert split for rationales
    if df_expert is not None:
        with st.spinner("Processing expert annotations for rationale extraction..."):
            df_expert["rationale_mask"] = [[0]*MAX_LEN for _ in range(len(df_expert))]
            expert_tokens = tokenize_batch(df_expert["raw_text"].tolist())
            
            progress_bar = st.progress(0)
            for idx in range(len(df_expert)):
                row = df_expert.iloc[idx]
                text = row["raw_text"]
                offsets = expert_tokens["offset_mapping"][idx].tolist()
                
                # Extract sentences using NLTK
                sentences = sent_tokenize(text)
                
                # Collect expert rationale sentences
                expert_sentences = set()
                for i in range(1, 6):
                    expert = row.get(f"expert_{i}")
                    if not expert:
                        continue
                    
                    # Handle different annotation formats
                    if isinstance(expert, str):
                        try:
                            # Try to parse as Python literal
                            expert = ast.literal_eval(expert)
                        except (ValueError, SyntaxError):
                            try:
                                # Try to parse as JSON
                                expert = json.loads(expert)
                            except json.JSONDecodeError:
                                # Skip if both parsing methods fail
                                continue
                
                # Extract sentences from expert annotations
                for rank in ['rank1', 'rank2', 'rank3', 'rank4', 'rank5']:
                    if rank in expert:
                        sentences_list = expert[rank]
                        if isinstance(sentences_list, str):
                            try:
                                # Parse string representation of list
                                sentences_list = ast.literal_eval(sentences_list)
                            except (ValueError, SyntaxError):
                                continue
                        if isinstance(sentences_list, list):
                            # Add cleaned sentences to the set
                            expert_sentences.update([s.strip() for s in sentences_list])
                
                # Create rationale mask
                mask = [0] * MAX_LEN
                for sent in sentences:
                    if sent.strip() in expert_sentences:
                        # Find all occurrences of the sentence
                        pattern = re.escape(sent)
                        for match in re.finditer(pattern, text):
                            start_idx = match.start()
                            end_idx = match.end()
                            
                            # Mark tokens within rationale span
                            for i, (start, end) in enumerate(offsets):
                                if i >= MAX_LEN:
                                    break
                                if start == 0 and end == 0:  # Skip special tokens
                                    continue
                                if not (end <= start_idx or start >= end_idx):
                                    mask[i] = 1
                
                df_expert.at[idx, "rationale_mask"] = mask
                progress_bar.progress((idx + 1) / len(df_expert))
            
            # Save expert data
            df_expert.to_csv("data/processed/expert.csv", index=False)
            torch.save(expert_tokens, "data/processed/expert_tokens.pt")
    
    # Save processed data
    OUTPUT_DIR = "data/processed"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df_train.to_csv(f"{OUTPUT_DIR}/train.csv", index=False)
    df_dev.to_csv(f"{OUTPUT_DIR}/dev.csv", index=False)
    df_test.to_csv(f"{OUTPUT_DIR}/test.csv", index=False)
    
    torch.save({
        "input_ids": train_tokens["input_ids"],
        "attention_mask": train_tokens["attention_mask"]
    }, f"{OUTPUT_DIR}/train_tokens.pt")
    
    torch.save({
        "input_ids": dev_tokens["input_ids"],
        "attention_mask": dev_tokens["attention_mask"],
        "offset_mapping": dev_tokens["offset_mapping"]
    }, f"{OUTPUT_DIR}/dev_tokens.pt")
    
    torch.save({
        "input_ids": test_tokens["input_ids"],
        "attention_mask": test_tokens["attention_mask"],
        "offset_mapping": test_tokens["offset_mapping"]
    }, f"{OUTPUT_DIR}/test_tokens.pt")
    
    # Calculate rationale coverage for expert split
    if df_expert is not None:
        def calculate_coverage(masks):
            total_tokens = 0
            positive_tokens = 0
            for m in masks:
                # Only consider actual text tokens (ignore padding)
                valid_tokens = len([x for x in m if x != -1])
                total_tokens += valid_tokens
                positive_tokens += sum(m[:valid_tokens])
            coverage = positive_tokens / total_tokens if total_tokens > 0 else 0
            return coverage
        
        expert_coverage = calculate_coverage(df_expert["rationale_mask"])
        st.success(f"Expert rationale coverage: {expert_coverage:.4%}")
    
    st.success(f"Preprocessing complete! Files saved to {OUTPUT_DIR}")
    
    return df_train, df_dev, df_test, df_expert if df_expert is not None else None

def distillation_training():
    # Configuration
    TEACHER_MODEL = "bert-base-uncased"
    STUDENT_MODEL = "distilbert-base-uncased"
    DATA_PATH = "data/processed/train_tokens.pt"
    SAVE_PATH = "models/distilled_model"
    BATCH_SIZE = 8
    EPOCHS = 1
    ALPHA, BETA, GAMMA = 0.5, 0.4, 0.1  # Adjusted loss weights
    
    # Load token data
    with st.spinner("Loading token data..."):
        token_data = torch.load(DATA_PATH, map_location='cpu')
    
    # Extract tensors
    input_ids = token_data["input_ids"]
    attention_mask = token_data["attention_mask"]
    
    # Create DataLoader
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize models
    with st.spinner("Loading teacher and student models..."):
        teacher = BertForMaskedLM.from_pretrained(TEACHER_MODEL).to(DEVICE)
        student = DistilBertForMaskedLM.from_pretrained(STUDENT_MODEL).to(DEVICE)
        teacher.eval()
    
    # Loss functions and optimizer
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer = AdamW(student.parameters(), lr=5e-5, weight_decay=1e-4)
    
    # Gradient accumulation steps
    grad_accum_steps = 2
    
    # Training loop
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart = st.line_chart(pd.DataFrame(columns=['Batch Loss', 'Avg Loss']))
    
    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            batch_input_ids, batch_attention_mask = [t.to(DEVICE) for t in batch]
            
            # Create masked inputs (15% masking probability)
            masked_input_ids = batch_input_ids.clone()
            mask_prob = torch.rand(masked_input_ids.shape, device=DEVICE)
            
            # Only mask non-special tokens
            special_tokens_mask = (batch_input_ids == 0) | (batch_input_ids == 101) | (batch_input_ids == 102)
            mask = (mask_prob < 0.15) & ~special_tokens_mask
            masked_input_ids[mask] = 103  # DistilBERT's mask token ID
            
            # Teacher forward pass
            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=masked_input_ids,
                    attention_mask=batch_attention_mask,
                    output_hidden_states=True
                )
            
            # Student forward pass with labels for proper MLM loss
            student_outputs = student(
                input_ids=masked_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=True,
                labels=batch_input_ids  # Add labels for built-in MLM loss
            )
            
            # Calculate losses
            # Use log probabilities for numerical stability
            student_log_probs = torch.nn.functional.log_softmax(student_outputs.logits, dim=-1)
            
            # Apply temperature scaling to teacher outputs
            teacher_logits = teacher_outputs.logits / 2.0
            teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
            
            distil_loss = kl_loss(student_log_probs, teacher_probs)
            
            # Hidden states loss
            student_hidden = student_outputs.hidden_states[-1]
            teacher_hidden = teacher_outputs.hidden_states[-1]
            
            # Flatten hidden states for cosine loss
            student_flat = student_hidden.view(-1, student_hidden.size(-1))
            teacher_flat = teacher_hidden.view(-1, teacher_hidden.size(-1))
            target = torch.ones(student_flat.size(0), device=DEVICE)
            
            cos_loss = cosine_loss(student_flat, teacher_flat, target)
            
            # Use built-in MLM loss (only calculates loss on masked tokens)
            mlm_loss = student_outputs.loss
            
            # Combined loss
            loss = (ALPHA * distil_loss + 
                    BETA * mlm_loss + 
                    GAMMA * cos_loss) / grad_accum_steps
            
            # Backpropagation with gradient accumulation
            loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum_steps
            
            # Update progress
            progress = (batch_idx + 1) / len(dataloader)
            progress_bar.progress(progress)
            
            # Update status
            batch_loss = loss.item() * grad_accum_steps
            avg_loss = total_loss / (batch_idx + 1)
            status_text.text(f"Epoch {epoch+1}/{EPOCHS} - Batch {batch_idx+1}/{len(dataloader)} - Loss: {batch_loss:.4f}, Avg Loss: {avg_loss:.4f}")
            
            # Update chart
            loss_chart.add_rows(pd.DataFrame({
                'Batch Loss': [batch_loss],
                'Avg Loss': [avg_loss]
            }, index=[batch_idx]))
            
            # Clear memory
            del masked_input_ids, mask_prob, mask, teacher_outputs, student_outputs
            torch.cuda.empty_cache()
            gc.collect()
        
        # Final gradient step if needed
        if len(dataloader) % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(dataloader)
        st.success(f"Epoch {epoch+1} Complete - Avg Loss: {avg_loss:.4f}")
    
    # Save distilled model
    os.makedirs(SAVE_PATH, exist_ok=True)
    student.save_pretrained(SAVE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    tokenizer.save_pretrained(SAVE_PATH)
    st.success(f"Distilled model saved to {SAVE_PATH}")

def train_classifier():
    # Configuration
    MODEL_PATH = "models/distilled_model"
    TRAIN_CSV = "data/processed/train.csv"
    DEV_CSV = "data/processed/dev.csv"
    SAVE_PATH = "models/classification_model"
    BATCH_SIZE = 8
    EPOCHS = 3
    LEARNING_RATE = 3e-5
    
    # Load data
    train_df = pd.read_csv(TRAIN_CSV)
    dev_df = pd.read_csv(DEV_CSV)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Try loading distilled model, fallback to pretrained if failed
    try:
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_PATH, 
            num_labels=2,
            ignore_mismatched_sizes=True
        ).to(DEVICE)
        st.info("Loaded distilled model for classification fine-tuning")
    except:
        st.warning("Failed to load distilled model, using pretrained as fallback")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=2
        ).to(DEVICE)
    
    # Create datasets and dataloaders
    train_dataset = ClassificationDataset(
        train_df["text"].tolist(), 
        train_df["label"].tolist(), 
        tokenizer
    )
    dev_dataset = ClassificationDataset(
        dev_df["text"].tolist(), 
        dev_df["label"].tolist(), 
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*EPOCHS)
    
    # Training loop
    best_f1 = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_chart = st.line_chart(pd.DataFrame(columns=['Train Loss', 'Val Loss', 'F1']))
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        epoch_progress = 0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(DEVICE)
            
            # Forward pass
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            epoch_progress = (batch_idx + 1) / len(train_loader)
            progress_bar.progress(epoch_progress)
            status_text.text(f"Epoch {epoch+1}/{EPOCHS} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in dev_loader:
                inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(DEVICE)
                
                outputs = model(**inputs, labels=labels)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                val_loss += outputs.loss.item()
        
        # Calculate metrics
        if len(all_labels) > 0:
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
            precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
            recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
            avg_val_loss = val_loss / len(dev_loader)
            
            # Update metrics chart
            metrics_chart.add_rows(pd.DataFrame({
                'Train Loss': [avg_train_loss],
                'Val Loss': [avg_val_loss],
                'F1': [f1]
            }, index=[epoch]))
            
            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                model.save_pretrained(SAVE_PATH)
                tokenizer.save_pretrained(SAVE_PATH)
                st.success(f"New best model saved to {SAVE_PATH} with F1: {f1:.4f}")
            
            st.info(f"Epoch {epoch+1} Evaluation:")
            st.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            st.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            st.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
        else:
            st.warning(f"Epoch {epoch+1} Evaluation: No valid labels to evaluate")
    
    st.success("Classification training complete!")

def train_rationale_model():
    # Configuration - ONLY USE EXPERT DATA FOR RATIONALE TRAINING
    MODEL_PATH = "models/distilled_model"
    EXPERT_CSV = "data/processed/expert.csv"
    SAVE_PATH = "models/rationale_model"
    BATCH_SIZE = 8
    EPOCHS = 10  # More epochs for small dataset
    LEARNING_RATE = 3e-5
    MAX_LEN = 512
    
    # Load expert data
    with st.spinner("Loading expert data for rationale training..."):
        expert_df = pd.read_csv(EXPERT_CSV)
    
    # Convert rationale masks
    expert_df["rationale_mask"] = expert_df["rationale_mask"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Split expert data into train and validation
    train_df, val_df = train_test_split(expert_df, test_size=0.2, random_state=42)
    
    # Enhanced class weight calculation
    def calculate_class_weights(masks):
        total_tokens = 0
        positive_tokens = 0
        for mask in masks:
            # Consider only first MAX_LEN tokens
            valid_mask = mask[:MAX_LEN]
            total_tokens += len(valid_mask)
            positive_tokens += sum(valid_mask)
        
        st.info(f"Positive tokens: {positive_tokens}/{total_tokens} ({positive_tokens/total_tokens:.4%})")
        
        # Handle case where there are no positive tokens
        if positive_tokens == 0:
            return torch.tensor([1.0, 1.0]).to(DEVICE)
        
        weight_positive = total_tokens / (2.0 * positive_tokens)
        weight_negative = total_tokens / (2.0 * (total_tokens - positive_tokens))
        
        st.info(f"Class weights - Negative: {weight_negative:.2f}, Positive: {weight_positive:.2f}")
        return torch.tensor([weight_negative, weight_positive]).to(DEVICE)
    
    # Calculate class weights using training data only
    class_weights = calculate_class_weights(train_df["rationale_mask"].tolist())
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Create datasets and dataloaders
    train_dataset = RationaleDataset(
        train_df["text"].tolist(),
        train_df["rationale_mask"].tolist(),
        tokenizer
    )
    val_dataset = RationaleDataset(
        val_df["text"].tolist(),
        val_df["rationale_mask"].tolist(),
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model from distilled base
    model = DistilBertForTokenClassification.from_pretrained(
        MODEL_PATH, 
        num_labels=2,
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*EPOCHS)
    
    # Loss function with class weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop
    best_f1 = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_chart = st.line_chart(pd.DataFrame(columns=['Train Loss', 'Val Loss', 'F1']))
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        epoch_progress = 0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = {
                'input_ids': batch['input_ids'].to(DEVICE),
                'attention_mask': batch['attention_mask'].to(DEVICE),
            }
            labels = batch['labels'].to(DEVICE)
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Calculate loss only on active tokens
            active_loss = inputs['attention_mask'].view(-1) == 1
            active_logits = logits.view(-1, 2)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            if active_labels.numel() > 0:
                loss = loss_fn(active_logits, active_labels)
            else:
                loss = torch.tensor(0.0, requires_grad=True).to(DEVICE)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            epoch_progress = (batch_idx + 1) / len(train_loader)
            progress_bar.progress(epoch_progress)
            status_text.text(f"Epoch {epoch+1}/{EPOCHS} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    'input_ids': batch['input_ids'].to(DEVICE),
                    'attention_mask': batch['attention_mask'].to(DEVICE),
                }
                labels = batch['labels'].to(DEVICE)
                
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Get predictions
                active_mask = inputs['attention_mask'].view(-1) == 1
                active_logits = logits.view(-1, 2)[active_mask]
                active_labels = labels.view(-1)[active_mask]
                
                preds = torch.argmax(active_logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(active_labels.cpu().numpy())
                val_loss += loss_fn(active_logits, active_labels).item() if active_labels.numel() > 0 else 0
        
        # Calculate metrics
        if len(all_labels) > 0:
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
            precision = precision_score(all_labels, all_preds, average="binary", zero_division=0)
            recall = recall_score(all_labels, all_preds, average="binary", zero_division=0)
            avg_val_loss = val_loss / len(val_loader)
            
            # Update metrics chart
            metrics_chart.add_rows(pd.DataFrame({
                'Train Loss': [avg_train_loss],
                'Val Loss': [avg_val_loss],
                'F1': [f1]
            }, index=[epoch]))
            
            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                model.save_pretrained(SAVE_PATH)
                tokenizer.save_pretrained(SAVE_PATH)
                st.success(f"New best model saved to {SAVE_PATH} with F1: {f1:.4f}")
            
            st.info(f"Epoch {epoch+1} Evaluation:")
            st.info(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            st.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        else:
            st.warning(f"Epoch {epoch+1} Evaluation: No valid labels to evaluate")
    
    # Always save final model
    if not os.path.exists(SAVE_PATH):
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        st.success(f"Final model saved to {SAVE_PATH}")
    
    st.success("Rationale training complete!")

# Streamlit UI
def main():
    st.title("⚖️ Legal Outcome Prediction System")
    st.markdown("""
    This system predicts legal case outcomes (ALLOWED/DISMISSED) and extracts key rationales using:
    - Knowledge distillation (BERT → DistilBERT)
    - Sequence classification for outcome prediction
    - Token classification for rationale extraction
    """)
    
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Mode", 
                               ["Demo", "Preprocess Data", "Distillation Training", 
                                "Classification Training", "Rationale Training"])
    
    if app_mode == "Demo":
        demo_mode()
    elif app_mode == "Preprocess Data":
        preprocess_mode()
    elif app_mode == "Distillation Training":
        distillation_mode()
    elif app_mode == "Classification Training":
        classification_mode()
    elif app_mode == "Rationale Training":
        rationale_mode()

def demo_mode():
    st.header("Case Outcome Prediction Demo")
    st.info("This demo predicts case outcomes and highlights key rationales")
    
    # Initialize inference system
    try:
        inference_system = InferenceSystem()
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.warning("Please train models first using the training options")
        return
    
    # Sample cases
    sample_cases = {
        "Murder Case": """
        The appellant was charged under Section 302 of the Indian Penal Code for the murder of his neighbor.
        Evidence shows the accused was present at the crime scene, and fingerprints match those found on the weapon.
        However, the defense argues there was no motive and the forensic evidence was mishandled by police.
        The prosecution maintains the circumstantial evidence is sufficient for conviction.
        """,
        "Contract Dispute": """
        The plaintiff claims breach of contract for non-payment of services rendered.
        The defendant acknowledges the services but disputes the quality and completeness.
        Emails between parties show agreement on deliverables and payment terms.
        The defendant refused payment citing unsatisfactory work.
        """,
        "Property Dispute": """
        The petitioner seeks declaration of title over ancestral property.
        Documents show the property was transferred to the respondent through a registered sale deed.
        The petitioner claims the transfer was fraudulent and without consent.
        Witnesses testify the petitioner was aware of the transaction.
        """
    }
    
    # Case selection
    case_option = st.selectbox("Choose a sample case or enter your own:", 
                             ["Select...", "Enter custom case"] + list(sample_cases.keys()))
    
    # Text input
    if case_option == "Enter custom case":
        text = st.text_area("Input Legal Case Text:", height=300, 
                          value="Enter case details here...")
    elif case_option != "Select...":
        text = st.text_area("Input Legal Case Text:", height=300, 
                          value=sample_cases[case_option])
    else:
        text = ""
        st.warning("Please select a sample case or choose 'Enter custom case'")
    
    # Prediction button
    if st.button("Predict Outcome and Rationale") and text.strip():
        with st.spinner("Analyzing case details..."):
            try:
                outcome, rationale = inference_system.predict(text)
                st.success(f"**Predicted Outcome:** {outcome}")
                
                st.subheader("Extracted Rationale")
                st.markdown("""
                <style>
                .rationale {
                    background-color: #f0f9ff;
                    padding: 15px;
                    border-radius: 10px;
                    border-left: 4px solid #1e88e5;
                    line-height: 1.8;
                    font-size: 16px;
                }
                </style>
                <div class="rationale">
                Key phrases are highlighted in brackets
                </div>
                """, unsafe_allow_html=True)
                
                # Display rationale with better formatting
                st.text(rationale)
                
                # Outcome explanation
                if outcome == "ALLOWED":
                    st.info("This case is likely to be ALLOWED - meaning the court will rule in favor of the petitioner/appellant")
                else:
                    st.info("This case is likely to be DISMISSED - meaning the court will rule against the petitioner/appellant")
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

def preprocess_mode():
    st.header("Data Preprocessing")
    st.info("This step prepares the legal dataset for training")
    
    if st.button("Run Data Preprocessing"):
        with st.spinner("Preprocessing data... This may take several minutes"):
            try:
                preprocess_data()
            except Exception as e:
                st.error(f"Preprocessing failed: {str(e)}")

def distillation_mode():
    st.header("Distillation Training")
    st.info("""
    Knowledge distillation process:
    1. Uses BERT-base as teacher model
    2. Trains DistilBERT as student model
    3. Combines KL divergence, cosine, and MLM losses
    """)
    
    if st.button("Start Distillation Training"):
        with st.spinner("Training distilled model... This may take 15-30 minutes"):
            try:
                distillation_training()
            except Exception as e:
                st.error(f"Distillation training failed: {str(e)}")

def classification_mode():
    st.header("Classification Model Training")
    st.info("Fine-tunes the distilled model for outcome prediction")
    
    if st.button("Start Classification Training"):
        with st.spinner("Training classification model... This may take 20-40 minutes"):
            try:
                train_classifier()
            except Exception as e:
                st.error(f"Classification training failed: {str(e)}")

def rationale_mode():
    st.header("Rationale Extraction Training")
    st.info("Trains model to identify key rationales using expert annotations")
    
    if st.button("Start Rationale Training"):
        with st.spinner("Training rationale model... This may take 30-60 minutes"):
            try:
                train_rationale_model()
            except Exception as e:
                st.error(f"Rationale training failed: {str(e)}")

if __name__ == "__main__":
    main()