# README #

This README would normally document whatever steps are necessary to get your application up and running.

### How to setup
- Clone the repository
- Create a virtual environment with `python -m venv venv`
- Activate the virtual environment
    ```
    source venv/Scripts/activate
    ```
- Run `pip install -r requirements.txt`
- Copy `.env.example` file and rename it into `.env`, Update all env variables like `OPENAI_API_KEY`

### How to run
- `uvicorn main:app --reload`
