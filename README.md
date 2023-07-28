# Bert_for_Question_and_Answering

This code performs fine-tuning on a pre-trained BERT-based model (DistilBERT) for question answering using the SQuAD v2 dataset. 

*Description of the model step by step:

Import Libraries and Set Seed: The code starts by importing necessary libraries, including PyTorch, Hugging Face Transformers, and the SQuAD dataset. It also sets a seed to ensure reproducibility.

Load the Dataset: The SQuAD v2 dataset is loaded using the load_dataset function. This dataset contains questions and corresponding contexts, along with the start and end positions of the answers within the contexts.

Select Random Subset: A random subset of 1000 examples is selected from the training set of the SQuAD dataset. This random selection is done without replacement.

Tokenization and Data Preparation: The code then proceeds with data preparation steps required for fine-tuning the model. The prepare_data function takes the selected examples, tokenizes the questions and contexts using the DistilBERT tokenizer, and prepares the inputs required for training. This includes truncating the input to a maximum length of 512 tokens, handling overflowing tokens, and determining the start and end positions of the answers within the tokenized inputs.

Data Loader Creation: The tokenized examples are converted into PyTorch DataLoader objects for efficient training. The custom_data_collator function groups the tokenized examples into batches and prepares the tensors for model input.

Model and Optimizer Initialization: The DistilBERT model for question answering is initialized with its tokenizer and configuration. An AdamW optimizer is set up with a learning rate of 3e-4.

Training Loop: The training loop runs for 6 epochs. In each epoch, the model is put into training mode (model.train()) and the selected examples are iterated in batches using the training DataLoader. For each batch, the model's outputs are computed and the loss is calculated based on the start and end position logits. The loss is backpropagated to update the model's parameters using the optimizer. Additionally, the training accuracy is calculated by comparing the predicted start and end positions with the true positions.

Evaluation: After each epoch, the model is switched to evaluation mode (model.eval()) and is tested on a sample question and context using the answer_question function. The predicted answer is printed for each epoch.

Plotting Loss and Accuracy: Finally, the code plots the average loss and accuracy per epoch using matplotlib.


**note

It's important to note that fine-tuning a language model requires substantial computational resources, and the provided code assumes the availability of a GPU (cuda) for accelerated training. If you don't have access to a GPU, you can remove the .to(device) calls to run the model on CPU (though the training process will be significantly slower).  Before running this code, you should ensure that all the required libraries (such as torch, datasets, transformers, etc.) are installed and that the squad_v2 dataset is properly loaded

** you can download weight of this  model from :
https://drive.google.com/file/d/1v15_isWL0UUMjx_8m8Bk_WHvTJYIfNyj/view?usp=drive_link

In summary, this code demonstrates how to fine-tune a DistilBERT model for question answering on a subset of the SQuAD v2 dataset. The model is trained through multiple epochs, and the average loss and accuracy are tracked during the training process. The evaluation of the model's performance on a sample question is also shown after each epoch. The plotted loss and accuracy curves provide insights into how the model's performance improves over training epochs.
