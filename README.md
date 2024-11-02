# Game of Thrones Q&A System

This project provides a question-and-answer system centered around the *Game of Thrones* books. Using LangChain, it loads a *Game of Thrones* PDF, tags scenes and characters, and leverages Hugging Face and FAISS to create a metadata-aware Q&A experience.

## Project Overview

- **`tagging.py`**: This file includes functions to load, split, and tag scenes and characters in the *Game of Thrones* PDF.
- **`got_qa_system.py`**: Contains the main `GoTQASystem` class, responsible for generating embeddings and managing the question-answer conversation chain.
- **`requirements.txt`**: Lists all necessary packages for the project.

## Setup Instructions

1. Clone the repository:

    ```bash
    git clone https://github.com/malekHattab/Game-of-Thrones-fans-chatbot-.git
    ```

2. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Add your Hugging Face API token to the `.env` file:

    ```plaintext
    HUGGINGFACEHUB_API_TOKEN=your_token_here
    ```

4. Place the `Game of Thrones.pdf` file in the main project directory.

## Running the Q&A System

To start the Q&A system, simply run:

```bash
python got_qa_system.py
```
