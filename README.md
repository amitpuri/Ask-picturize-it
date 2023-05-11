# Ask-me-to-picturize-it

[![licenses](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Huggingface Space 🤗

[https://huggingface.co/spaces/amitpuri/Ask-picturize-it](https://huggingface.co/spaces/amitpuri/Ask-picturize-it)

[![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/amitpuri/Ask-me-to-picturize-it/blob/main/app.ipynb)

This uses OpenAI API Whisper(whisper-1), DALL-E, GPT(gpt-3.5-turbo)

Extracted from https://openai.com/pricing

Multiple models, each with different capabilities and price points. Prices are per 1,000 tokens. 
You can think of tokens as pieces of words, where 1,000 tokens is about 750 words. 
This paragraph is 35 tokens.

GPT-4

With broad general knowledge and domain expertise, 
GPT-4 can follow complex instructions in natural language and solve difficult problems with accuracy.

| Model         | Prompt            | Completion               |
|---------------|-------------------|--------------------------|
| 8K context    | $0.03 / 1K tokens | $0.06 / 1K tokens        |
| 32K context   | $0.06 / 1K tokens | $0.12 / 1K tokens        |


| Model            | Usage             | 
|------------------|-------------------|
| gpt-3.5-turbo    | $0.002 / 1K tokens| 

--------------------------------------------------------

DALL-E

| Resolution            | Price        | 
|------------------|-------------------|
| 1024×1024        | $0.020 / image    | 
| 512×512          | $0.018 / image    | 
| 256×256          | $0.016 / image    | 

--------------------------------------------------------

Audio 

| Model            | Usage                                          | 
|------------------|------------------------------------------------|
| Whisper          | $0.006 / minute (rounded to the nearest second)| 


--------------------------------------------------------
Fine-tuning models
Create your own custom models by fine-tuning our base models with your training data. Once you fine-tune a model, 
you’ll be billed only for the tokens you use in requests to that model.

| Model         | Training            | Usage                  |
|---------------|---------------------|------------------------|
| Ada           | $0.0004 / 1K tokens | $0.0016 / 1K tokens    |
| Babbage       | $0.0006 / 1K tokens | $0.0024 / 1K tokens    |
| Curie         | $0.0030 / 1K tokens | $0.0120 / 1K tokens    |
| Davinci       | $0.0300 / 1K tokens | $0.1200 / 1K tokens    |


--------------------------------------------------------
Embedding models
Build advanced search, clustering, topic modeling, and classification functionality with our embeddings offering.

| Model        | Usage                 | 
|--------------|-----------------------|
| Ada          | $0.0004 / 1K tokens   | 


    docker build --rm --pull \
      --file "Dockerfile" \
      --label "com.amitpuri.ask-picturize-it" \
      --tag "ask-picturize-it:latest" \
      .

    docker run -e P_MONGODB_DATABASE -e P_MONGODB_URI -it --publish 80:80 --publish 27017:27017 ask-picturize-it:latest
