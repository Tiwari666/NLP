# Natural Language Processing (NLP) 
NLP is a subfield of artificial intelligence (AI) that focuses on the interaction between computers and human languages. The goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and contextually relevant.

# Application of NLP:
1) *Text Classification:** NLP algorithms are employed for text classification tasks, such as email filtration :spam detection, fishing attempts, sentiment analysis, and topic categorization.

2) *Sentiment Analysis**
Sentiment analysis is the process of analyzing emotions within a text and classifying them as positive, negative, or neutral. By running sentiment analysis on social media posts, product reviews, NPS surveys, and customer feedback, businesses can gain valuable insights about how customers perceive their brand.

3) *Machine Translation**:The automatic translation of text from one language to another is a classic NLP problem. Google Translate is a common example of a machine translation system.

4). **Chatbots:** Businesses use NLP-powered chatbots to provide automated customer support.

5) *SPEECH RECOGNITION**: aLEXA, SIRI,CORTANA, GOOGLE ASSISTANT.

6) **Financial Analysis:** NLP helps analyze financial news, reports, and social media to gain insights into market trends, sentiments, and investment opportunities.

# 3-part series on Natural Language Processing (NLP)

A) NLP Part 1: Scraping the Web using BeautifulSoup and Python

B) NLP Part 2: Pre-Processing Text Data Using Python:
Text are preprossed using different python frameworks (libraries) such as
NLTK, SpaCy,.


C) NLP Part 3 : Exploratory Data Analysis of Text Data:
As usual using the ML algorithms, WE CAN DO THE EXPLORATORY DATA ANALYSIS....



Overall, NLP is still at a primitive stage.
It is sttill expanding AND HIGLY EMERGING FIELD.
BASED ON THE NLP, various kinds of LLM are developed...


LINK FOR PREPROCESSING: https://spotintelligence.com/2022/12/21/nltk-preprocessing-pipeline/

# Steps in NLP

1) Set up the environment: Cloud account(AWS, GCP...), Jupyter Notebook,....

2) Install and import relevant libraries: nltk, re,spacy

3) Web scrape through the Beatiful Soup Library/ Load the data

4) Preprocess the data: remove non-alphabetic tokens--Remove noisy data

Top 14 NLTK preprocessing steps

4.1. Tokenization: Tokenization involves breaking down a word or a sentence or a paragraph into smaller units, typically words or subwords, to enable further analysis and processing.

4.2. Lowercasing

4.3. Remove punctuation

4.4 Remove stop words: dictionary

4.5. Remove extra whitespace

4.6. Remove URLs

4.7. Remove HTML code

4.8. Remove frequent words

4.9. Spelling correction

4.10. Stemming: Stemming is the process of reducing words to their root or base form, typically by removing suffixes and sometimes prefixes, with the aim of achieving linguistic normalization and improving text analysis in natural language processing tasks.


4.11. Lemmatization : Lemmatization is the process of reducing words to their base or dictionary form (lemma) while considering the word's meaning and context. Unlike stemming, which simply chops off prefixes or suffixes, lemmatization involves morphological analysis to accurately return the base or dictionary form of a word.

The main difference between stemming and lemmatization is that stemming may result in a root form that is not an actual word, whereas lemmatization always returns a valid word. For example, stemming might convert "running" to "run", whereas lemmatization would convert it to "run" as well, ensuring it remains a valid word form. Lemmatization is often considered more sophisticated and accurate than stemming, but it can also be computationally more expensive.


4.12. Part-of-speech tagging: Part-of-speech tagging, also known as POS tagging or grammatical tagging, is the process of assigning grammatical tags (such as noun, verb, adjective, etc.) to each word in a text based on its syntactic role within the sentence. 

4.13 Named Entity Recognition : Named Entity Recognition (NER) is a natural language processing task that involves identifying and categorizing named entities (such as names of people, organizations, locations, dates, etc.) within a text. The goal of NER is to accurately locate and classify these entities to extract useful information and improve understanding of the text's content.


4.14. Normalization: Normalize the US or UK English



5) Modeling :
A unique index or ID is assigned after the preprocessing of the data, which help to vectorize the text under the idea of Word Embedding via the
Word2Vec models, such as Continuous Bag of Words (CBOW) and Skip-gram.
Word2Vec models, such as Continuous Bag of Words (CBOW) and Skip-gram, learn to predict the context of a word given its neighboring words or vice versa, thereby generating dense, semantic embeddings for each word in the vocabulary.

Alternatively, use the GloVe algorithm to generate word embeddings. GloVe (Global Vectors for Word Representation) is based on matrix factorization techniques and leverages global word-word co-occurrence statistics to learn word embeddings.

6) Model Hyperparameter tuning

7) Prediction

8) Model Deployment to production


# Two different approaches for Text Summarization
A) Extractive Summarization

B) Abstractive Summarization

A) Extractive Summarization:

In Extractive Summarization, we identify essential phrases or sentences from the original text and extract only these phrases from the text. These extracted sentences would be the summary.

BERTSUM: While BERT (Bidirectional Encoder Representations from Transformers) is primarily used for abstractive summarization, BERTSUM is a variant that fine-tunes BERT for extractive summarization by scoring sentences based on their representation in the BERT model.

LSTM-based models: Long Short-Term Memory (LSTM) networks and their variants, such as Bidirectional LSTMs (BiLSTMs) or Gated Recurrent Units (GRUs), can be adapted for extractive summarization tasks by training them to predict the importance of each sentence in the document.

One of the most frequently used models for extractive summarization is the TextRank algorithm. TextRank is a graph-based ranking algorithm inspired by Google's PageRank algorithm, which ranks web pages based on their importance. In TextRank, a graph is constructed where nodes represent sentences, and edges between nodes represent the similarity between sentences.

B) Abstractive Summarization:

We work on generating new sentences from the original text in the Abstractive Summarization approach. The abstractive method contrasts the approach described above, and the sentences generated through this approach might not even be present in the original text.

Deep Learning Models (e.g., Transformer-based models like BERT, GPT) come under the Abstractive Summarization.

Sequence-to-Sequence Models (e.g., LSTM, GRU) come under the Abstractive Summarization.


# Note on NLTK:

NLTK (Natural Language Toolkit) is a comprehensive library for natural language processing tasks in Python. It provides various tools and resources for a wide range of tasks, including tokenization, stemming, lemmatization, part-of-speech tagging, and more.

However, NLTK itself does not perform abstractive or extractive summarization out of the box. Instead, NLTK provides the building blocks and utilities that can be used to implement summarization algorithms, but one would need to write the code or use additional libraries to perform the actual summarization task.

For example, NLTK provides functions for text preprocessing (e.g., tokenization) and linguistic analysis (e.g., part-of-speech tagging), which can be useful in developing summarization systems. But for abstractive or extractive summarization, one would typically use NLTK in conjunction with other libraries or algorithms specifically designed for summarization tasks, such as TextRank, BERT, or other machine learning models.

So, to perform abstractive or extractive summarization using NLTK, one would typically use NLTK for text preprocessing and linguistic analysis, and then implement or integrate algorithms or models for the summarization task itself.




# Terms Used in NLP:

# A) Corpus:

A collection of text is known as Corpus. This could be data sets such as bodies of work by an author, poems by a particular poet, etc. To explain this concept in the blog, we will use a data set of predetermined stop words.

# B) Tokenizers:

Tokenization is the process of breaking down a stream of text into smaller units called tokens i.e., This divides a text into a series of tokens.

Tokens serve as the basic building blocks for various NLP tasks such as parsing, sentiment analysis, part-of-speech tagging, and named entity recognition.



Tokenizers have three primary tokens – sentence, word, and regex tokenizer. We will be using only the word and the sentence tokenizer.

# C) TF-IDF (Term Frequency-Inverse Document Frequency):

This method assigns weights to each term in a document based on its frequency in the document and its rarity in the entire corpus. Sentences with the highest TF-IDF scores are selected for the summary under the Extractive Summarization technique.

#D) Word embedding
Word embeding is a representation of words in a continuous vector space where words with similar meanings are mapped to nearby points. In other words, it's a mathematical technique to represent words as vectors (arrays of real numbers) in such a way that the geometric distance between these vectors captures the semantic similarity between the corresponding words.

Word embeddings are typically learned from large corpora of text data using techniques like Word2Vec, GloVe (Global Vectors for Word Representation), or embeddings from pre-trained language models like BERT (Bidirectional Encoder Representations from Transformers).


