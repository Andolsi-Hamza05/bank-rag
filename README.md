# bank-rag
A comprehensive application for chatting with banking financial documents

# My Project

## Setup Instructions

### Prerequisites

- Ensure you have [Poetry](https://python-poetry.org/docs/#installation) installed.

### Steps to Clone and Set Up the Project

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Andolsi-Hamza05/bank-rag.git
    cd <repository>
    ```

2. **Install project dependencies**:
    ```sh
    poetry install
    ```

3. **Activate the virtual environment**:
    ```sh
    poetry shell
    ```

4. **Configure ngrok**:
    - Ensure you have an ngrok account. If you don't have one, sign up at [ngrok](https://ngrok.com/).
    - Once you have an account, get your authtoken from the ngrok dashboard.
    - Add the authtoken to your ngrok configuration:
        ```sh
        ngrok config add-authtoken <YOUR_NGROK_AUTH_TOKEN>
        ```

5. **Set up environment variables**:
    - Create a `.venv` file in the root directory of the project.
    - Add the `HUGGINGFACEHUB_API_TOKEN` variable to the `.venv` file:
        ```
        HUGGINGFACEHUB_API_TOKEN=<YOUR_HUGGINGFACEHUB_API_TOKEN>
        ```

6. **Run the project**:
    - Start the FastAPI server:
        ```sh
        python src/apis/main.py
        ```

    - In the terminal, you will see a line similar to:
        ```
        Public URL: https://<ngrok_generated_url>.ngrok.io
        ```

7. **Configure Copilot Studio**:
    - Copy the printed ngrok URL from the terminal.
    - Add `/chat/` to the end of the URL.
    - In Copilot Studio, go to the Conversational Boosting section.
    - Replace the URL in the HTTP request field with the complete ngrok URL (e.g., `https://<ngrok_generated_url>.ngrok.io/chat/`).

8. **Using Copilot Studio as your UI**:
    - You can now use Copilot Studio to interact with the application and chat with your banking financial documents.

