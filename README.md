# Marketing Planner Agent

This project is a Streamlit web application that provides a chat interface to a marketing planner agent. The agent is powered by Google's Gemini model and helps users create marketing plans by gathering necessary information through a conversation.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2.  Navigate to the project directory:
    ```bash
    cd <project-directory>
    ```
3.  Install the required Python libraries from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

This application requires a Bearer Token and a Google API key to function correctly.

1.  **Bearer Token**: The application needs a Bearer Token to fetch available models from the API. This token needs to be refreshed every hour.

2.  **Google API Key**: A Google API key with access to the Gemini API is required.

You can configure these keys by creating a `.env` file in the root of the project directory:

```
BEARER_TOKEN="your_bearer_token_here"
GOOGLE_API_KEY="your_gemini_api_key_here"
```

## Usage

To run the application, use the following command in your terminal:

```bash
streamlit run app.py
```

This will start the Streamlit server and open the application in your web browser. You can then interact with the marketing planner agent through the chat interface.
