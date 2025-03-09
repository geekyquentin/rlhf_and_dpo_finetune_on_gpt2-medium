# Implementing Reinforcement Learning with Human Feedback (RLHF) and Direct Preference Optimization (DPO) to generate non-harmful answers when prompted with harmful and stereotypical questions.

The train and test datasets, the trained models (reward model, rlhf trained and ppo trained models), and the code are available in the kaggle notebook: https://www.kaggle.com/code/yashwantkrishna/genai

# Requirements
```
numpy==2.2.3
pandas==2.2.3
rouge-score==0.1.2
torch== 2.6.0
tqdm==4.67.1
transformers==4.49.0
trl==0.9.4
```

# Task A: Implementing RLHF
1. We use BertTokenizer from `bert-base-uncased` to process text data.
2. The PreferenceDataset class loads preference data from a CSV file, with max_length set to 512.
3. Designed `RewardModel` class.
  - BERT-based model with a fully connected (fc) layer to output scores.
4. Used Bradley-Terry loss to compare scores of more and less preferred responses.
5. Loaded dataset with batch size of 8.
6. Trained for 3 epochs using AdamW optimizer with lr=5e-5.

## Using PPO to fine-tune
1. GPTTokenizer: for policy and reference models.
2. BertTokenizer: for using reward model.
3. PPO Trianer's config: 5e-6, batch size: 2, mini-batch size: 2.
4. Load the dataset with `DataLoader`,  where the PreferenceDataset's IDs have only 256 tokens.
5. Iterate over the dataset for 1 epoch.
6. The policy model generates responses with a maximum of 30 tokens.
7. Decode the responses and tokenize generated responses with BertTokenizer for reward evaluation.
8. Use the reward model to compute rewards for each response.
9. PPOTrainer's step: queries, responses, and rewards as parameters.

# Task B: Implementing DPO
1. Define `DPOModel`, which loads 'GPT-2 Medium' as the base model.
2. Set the precision to `float32` so that the loss doesn't go to `inf` or `nan`.
3. Loss function is calculated as given, along with clamping extreme values to prevent `inf `or `nan`.
4. Load the dataset with `DataLoader` with a batch size of 2.
5. Initialize DPOModel for training and another model as a frozen reference.
6. Iterate over the dataset for 3 epochs.
7. Extract tokenized inputs for preferred and less-preferred responses.
8. Compute logits using the trainable model and the frozen reference model.
9. Calculate loss and perform backpropagation.

# Task C: Performance Comparison
1. Load `rlhf_trained` and `dpo_trained`.
2. Define `DPOModelForEval`, a subclass of `DPOModel` to override the `generate()` function.
3. Load trained weights for `DPOModelForEval`.
4. Load the testset.
5. Evaluation metrics used: BLEU and ROUGE (ROUGE-1, ROUGE-2, and ROUGE-L).
6. `generate()` produces responses, which are decoded and compared against the reference answers.

