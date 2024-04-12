# Transformer Model with Switch FFN 🚀

Welcome to this repository where I have implemented a Encoder only Transformer with a Switch Feed Forward Network (FFN) layer!

## Dataset Preprocessing 🧹

- Loaded the Wikitext dataset from two parquet files and combined them into a single DataFrame.
- Tranformed that DataFrame into one long string for processing.
- I wanted to train our own tokenizer, but ran into some loading issues. So, I decided to use a pre-trained tokenizer from the Hugging Face library instead. 🤗
- Custom tokenizer is WIP
- To keep things speedy for testing, I only used 1% of the data for training and the next 0.5% for validation.

## Model Architecture 🏗️

Started with a standard implementation of the Transformer encoder, just to get a feel for things. It's got all the usual components:

- An embedding layer that handles token and position embeddings
- An encoder with multiple blocks, each containing a multi-head attention layer and a feed-forward network (FFN)
- Layer Normalization and dropout for regularization

After that the Switch FFN is Introduced:

- The Switch FFN routes tokens to different experts based on a linear gating mechanism.
- Gating mechanism with attention is WIP.
- I have added extra parameters like the number of experts, top-k selection, and expert dropout to really customize the Switch FFN.
- The switch transformer has a custom loss function as well, called the load balancing loss function. It makes sure the tokens are routed to all experts appropriately.
- The switch layer is introduced afetr every 3 standard encoder FFN, same as the paper.

Implementing an encoder with Mixture of Experts (MoE) is pretty uncommon. Mostly it is decoder only MoE's. Excited to test it out on different down stream tasks and get to the HF leaderboard with it. :)

## Implementation Details 🔍

I have buiult this using PyTorch, and was super aware about keeping track of tensor shapes. I have shape comments at every computation to make sure we know what's going on with inputs and outputs.

I tested each part of the model with random tensors just to double-check that the  outputs were correct.

## Training and Evaluation 📈

For training, I used the masked language modeling objective. Essentially, we masked some of the input tokens and let the model try to predict the original ones. Fill- in-the-blanks.

I processed the training data in chunks to keep things fast and used a custom dataset and data loader.

## Usage 🚀

It has a requirements.txt file. Since it is implemented from scratch only 3 depenedencies. pandas, torch and transformers (for the tokenizer only)

## Acknowledgments 🙌

I couldn't have done this without these papers:

- The Transformer architecture from the "Attention Is All You Need" paper by Vaswani et al.
- The Switch FFN layer, inspired by the "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" paper by Fedus et al.

