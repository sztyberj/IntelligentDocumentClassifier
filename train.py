from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from src.data_processing.data_prep import DataPrep

#1. Init DataPrep and Config
data_prep = DataPrep(config_path="./config.yml")
config = data_prep.get_config()

MODEL_NAME = config["model"]["name"]
MAX_SEQ_LENGHT = config["model"]["max_seq_length"]
BATCH_SIZE = config["train"]["batch_size"]
EPOCHS = config["train"]["epochs"]
WARMUP_STEPS = config["train"]["warmup_steps"]
OUTPUT_PATH = config["train"]["output_path"]

#2. Load data
print("--- 1. Loading Data ---")
train_examples = data_prep.create_training_examples("./data")

if not train_examples:
    print("ERROR: 0 training data")
    exit()

#3. Model init
print(f"\n--- 2. Initializing Model: {MODEL_NAME} ---")
model = SentenceTransformer(MODEL_NAME)
model.max_seq_lenght = MAX_SEQ_LENGHT

#4. Prepare DataLoader and Loss
train_data_loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model)

#5. Training
print(f"\n--- 3. Starting Training on GPU (Batch: {BATCH_SIZE}) ---")

model.fit(
    train_objectives=[(train_data_loader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=WARMUP_STEPS,
    show_progress_bar=True,
    use_amp=True
)

#6. Save model
print(f"\n--- 4. Saving Model to {OUTPUT_PATH} ---")
model.save(OUTPUT_PATH)
print("✅ Training END")