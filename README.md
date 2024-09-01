# Dynamic Return Policy API

This project provides an API to generate dynamic return policies for different product categories based on customer trustworthiness scores. The API is built using FastAPI and integrates with HuggingFace models using LangChain to provide AI-generated return policies.

## Installation

1. Clone the Repository

```
git clone https://github.com/Manvith-Shetty/Return-Policy.git
cd Return-Policy
```

2. Create a Virtual Environment

```
python -m venv venv
venv\Scripts\activate # on linux source venv/bin/activate
```

3. Install Dependencies

```
pip install -r requirements.txt
```

4. Set up Environment Variables

```
HF_TOKEN=your_huggingface_api_token
```

## Running the Application

Start the FastAPI Server

```
fastapi run app.py --port 8000 --reload
```


