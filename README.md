# Deep-Learning-Training

CHAPTER – 1 

Understanding how computers process language 

 

NLP – natural language processing 

PYTHON can store a sentence as a text, but a neural network generally needs numerical input. 

We must convert the sentence into a representation that the model can process 

The process begins with tokenization. 

Tokenization beaks text into smaller units called tokens. 

Depending on the tokenizer, a token can be complete word, part of a word , punctuation, or another text unit. 

A Neural network does not normally work directly with words. We need to assign each token an integer ID. 

 

Suppose our small vocabulary is : 

Token                                            Token ID 

I                                                        1    

Love                                                2 

Deep                                               3             

Learning                                        4 

 

 

This ID are identifiers,  not numerical measures of meaning. 

Token ID 4 is not necessarily more meaningful than token ID 1. 

 

Practical 2: Create a vocabulary 

  

What is happening internally: 

The dictionary stores a mapping from each word to an integer. The list comprehension looks up the ID for every word in the original sentence. 

  

In a real application, the vocabulary is established by a tokenizer and may contain thousands or hundreds of thousands of tokens. Special tokens may also represent padding, unknown words, or boundaries. 

  

Important limitation: Our vocabulary only contains words from this sentence. If we try to look up a new word, such as "AI", it will fail unless we add it or handle unknown tokens. Real tokenizers solve this in more sophisticated ways. 

 

Lesson 1.4: What are embeddings? 

Now we reach one of the most important concepts in modern NLP. 

  

Tell students: 

  

"We have converted words into IDs. But an ID alone does not tell the neural network much about the meaning of a word. We need a representation that the model can learn from. That representation is called an embedding." 

  

Understand embeddings through an example 

Imagine we have these words: 

- king 

- queen 

- apple 

- banana 

  

We want the model to learn useful patterns about how these words are used. 

An embedding maps each token ID to a vector of numbers. 

 For example, imagine the following simplified vectors: 
 
 
 

 

 

We want the model to learn useful patterns about how these words are used. 

 An embedding maps each token ID to a vector of numbers. 

For example, imagine the following simplified vectors:  

Word| Dimension 1| Dimension 2| Dimension 3 

king| 0.8| 0.6| 0.2 

queen| 0.7| 0.7| 0.2 

apple| 0.1| 0.2| 0.9 

banana| 0.2| 0.3| 0.8 

  

These values are invented for illustration. In a real model, embedding dimensions are learned from data and usually do not have such straightforward, human-readable meanings. 

The important difference is:  

- Token ID: Identifies a token. 

- Embedding: Represents that token using a vector of learned numerical values. 

  

Practical 3: Create Embeddings Using TensorFlow 

The actual embedding values will be initialized by TensorFlow and will vary. 

Explain the output shape 

The output shape is (1, 4, 3). 

- 1 means one sentence in the batch. 

- 4 means four tokens in the sentence. 

- 3 means each token has a three-dimensional embedding. 

Understand the tensor shape 

Value| Meaning 

1| Batch size 

4| Sequence length 

3| Embedding size 

 Lesson 1.5: Why Do We Need Attention? 

  

Now explain the problem that motivates attention.  

Consider these two sentences: 

Sentence A: "The animal didn't cross the road because it was tired." 

Sentence B: "The animal didn't cross the road because it was busy." 

The meaning of a word depends on its context. To interpret a sentence, the model needs to build relationships between words. 

Older sequence models, such as recurrent neural networks (RNNs), process a sequence step by step. Although they can learn context, maintaining useful information over long sequences can be challenging. 

 

This idea is the foundation of the Transformer architecture. 

  

------------------------------------------------------------ 

Chapter 2 – The attention mechanism, step by step 

  

Lesson 2.1: Understand Query, Key and Value 

Before showing the formula, use an everyday example. 

Imagine a student searching a digital library for information about deep learning. 

- The student has a query: "I want to learn attention mechanisms." 

- Each document has a key: information that helps match it to a search. 

- Each document has a value: the actual information that can be retrieved. 

The search system compares the query with the keys, determines relevance weights, and 

combines information from the values. 

This is an analogy, not a literal description of how every search engine works.  

What are Q, K and V mathematically? 

In a Transformer, the input token representations are transformed using learned 

Understand the code  

Step 1 – define Q 

Each row is query vector . We have three queries , each with two dimensions  

Step 2 – define K 

Each row is a key vector . We use three keys , each with two dimensions  

Step 3 – define V 

Each row is a value vector . Notice that these values are delibrately different from the query and key vector . This helps demonstrate that scores determine how values are combined 

Step 4 – calculate scores  

Scores = tf.matmul(Q,K,transpose_b=true) 

tf.matmul performs matrix multiplication. Setting transpose_b=True transposes the second matrix for the calculation. 

The result has shape (3, 3): three queries compared with three keys. 

Step 5 – Scale the scores 

scaled_scores = scores / tf.math.sqrt(key_dimension) 

We divide by the square root of the key dimension, which is 2 in this example. 

Step 6 – Calculate weights 

weights = tf.nn.softmax(scaled_scores, axis=-1) 

axis=-1 tells TensorFlow to apply softmax across the final dimension, meaning across each row. 
 
 
CHAPTER 3 

What exactly is transformer  

A transformer is a neural network architechture that uses attention and feed forward networks to process sequences 

 

Self attention 

Self attention allows each token to calculate how it relates to other tokens in the same sequences4 
