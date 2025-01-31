import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
def load_custom_dataset(file_path):
    dataset = load_dataset('text', data_files=file_path)
    return dataset

# Tokenize dataset
def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

# Main function
def main():
    # Load pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Load and tokenize custom dataset
    dataset = load_custom_dataset('data/train.txt')
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=500,
        prediction_loss_only=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("./fine_tuned_gpt2")
    tokenizer.save_pretrained("./fine_tuned_gpt2")

    # Test the model
    input_text = "Once upon a time"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
