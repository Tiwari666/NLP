#Natural Language Processing (NLP) is a subfield of artificial intelligence (AI) that focuses on the interaction between computers and human languages. The goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and contextually relevant.

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
#Steps in NLP

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



6) Modeling :
A unique index or ID is assigned after the preprocessing of the data, which help to vectorize the text under the idea of Word Embedding via the
Word2Vec models, such as Continuous Bag of Words (CBOW) and Skip-gram.
Word2Vec models, such as Continuous Bag of Words (CBOW) and Skip-gram, learn to predict the context of a word given its neighboring words or vice versa, thereby generating dense, semantic embeddings for each word in the vocabulary.

Alternatively, use the GloVe algorithm to generate word embeddings. GloVe (Global Vectors for Word Representation) is based on matrix factorization techniques and leverages global word-word co-occurrence statistics to learn word embeddings.

7) Model Hyperparameter tuning

8) Prediction

9) Model Deployment to production

