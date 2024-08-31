from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()

sec_key = os.getenv('HF_TOKEN')

app = FastAPI()

#Cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/hello")
async def hello():
    return "welcome"

@app.post("/general")
async def general(customer_id: str, customer_score: float):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=700, temperature=0.8, huggingfacehub_api_token=sec_key)
    #prompt template
    general_question = """Generate a single general return policy for a customer {customer_id} with a trustworthiness score of {customer_score} out of 100. Include a return window (in days), condition requirements, and any special notes. Be stricter for lower scores and more lenient for higher scores like Longer return window and lenient condition requirements for high score customers. Dont exceed 30 days.Format the output as:
    Return Window: [X days]
    Condition Requirements: [requirements]
    Special Notes: [notes]"""

    general_policy_prompt_template = PromptTemplate(template=general_question, input_variables=["customer_id", "customer_score"])

    general_policy_chain = LLMChain(llm=llm, prompt = general_policy_prompt_template)

    raw_general_policy = general_policy_chain.invoke({"customer_id":customer_id, "customer_score":customer_score})

    # Parse the raw general policy
    lines = raw_general_policy['text'].strip().split('\n')
    general_return_window = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Return Window')), "14 days")
    general_condition = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition Requirements')), "Item must be in original condition")
    general_special_notes = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Special Notes')), "No special notes")

    return {"Return_Window": general_return_window, "Condition_requirements": general_condition, "Special_notes": general_special_notes}



@app.post("/tv")
async def electronics(customer_id: str, customer_score: float):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=700, temperature=0.8, huggingfacehub_api_token=sec_key)

    category_question = """Generate a specific return policy for {category} products for a customer {customer_id} with a trustworthiness score of {customer_score} out of 100. The policy should be lenient for higher scores and stricter for lower scores. Provide the details directly, without headings or redundant information. Limit the response to 3-5 sentences. Ensure the style is consistent across all categories. Format the output as below:
    Pay on delivery: Whether Available [yes] or [no] for that customer and explain about it.
    Returnable: Whether available [yes] or [no] for that customer. If [Yes] then only give the below Return Window for that customer. For electronics don't give return for low customer scores, instead give replacement for genuine cases.
    Return Window: Specify the time frame within which returns are accepted for that customer.
    Condition of Items: Mention the condition in which items must be returned for that customer.
    Exceptions and Restrictions: Highlight any exceptions or restrictions that apply for that customer.
    Refunds and Exchanges: State the policy on refunds and exchanges, including who covers return shipping costs if applicable for that customer.
    Additional Notes: Include any additional notes relevant to the return policy for that customer.
    """
    category_policy_prompt_template = PromptTemplate(template=category_question, input_variables=["customer_score", "customer_id", "category"])
    category_policy_chain = LLMChain(llm=llm, prompt = category_policy_prompt_template)

    response = category_policy_chain.invoke({"customer_score": customer_score, "customer_id": customer_id, "category": "Tv, Appliances, Electronics"})

    electronics_policy = response['text']

    lines = electronics_policy.strip("-").strip().split('\n')
    pay_on_delivery = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Pay on delivery')), "Yes, this is available {customer_id}")
    returnable = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Returnable')), "Yes, the customer can return electronics.")
    return_window = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Return Window')), "14 days")
    condition_of_items = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "The items must be in their original packaging and unused for a return to be accepted.")
    exceptions = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "Items returned after the 14-day window may not be eligible for a refund, but we will provide a replacement at no additional cost.")
    refunds_exchanges = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "We will process refunds within 3-5 business days of receiving the returned item.")
    additional_notes = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "For electronics, we recommend checking the product thoroughly upon delivery to ensure it's in working order before signing for it. If you notice any issues, please contact us immediately.")

    return {"Pay_on_delivery": pay_on_delivery, "Returnable": returnable, "Return_window": return_window, "Condition_of_items": condition_of_items, "Exceptions": exceptions, "Refunds_exchanges": refunds_exchanges, "additional_notes": additional_notes}

@app.post("/fashion")
async def fashion(customer_id: str, customer_score: float):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=700, temperature=0.8, huggingfacehub_api_token=sec_key)

    category_question = """Generate a specific return policy for {category} products for a customer {customer_id} with a trustworthiness score of {customer_score} out of 100. The policy should be lenient for higher scores and stricter for lower scores. Provide the details directly, without headings or redundant information. Limit the response to 3-5 sentences. Ensure the style is consistent across all categories. Format the output as below:
    Pay on delivery: Whether Available [yes] or [no] for that customer and explain about it.
    Returnable: Whether available [yes] or [no] for that customer. If [Yes] then only give the below Return Window for that customer.
    Return Window: Specify the time frame within which returns are accepted for that customer.
    Condition of Items: Mention the condition in which items must be returned for that customer.
    Exceptions and Restrictions: Highlight any exceptions or restrictions that apply for that customer.
    Refunds and Exchanges: State the policy on refunds and exchanges, including who covers return shipping costs if applicable for that customer.
    Additional Notes: Include any additional notes relevant to the return policy for that customer.
    """
    category_policy_prompt_template = PromptTemplate(template=category_question, input_variables=["customer_score", "customer_id", "category"])
    category_policy_chain = LLMChain(llm=llm, prompt = category_policy_prompt_template)

    response = category_policy_chain.invoke({"customer_score": customer_score, "customer_id": customer_id, "category": "fashion"})

    fashion_policy = response['text']
    print(fashion_policy)

    lines = fashion_policy.strip('-').strip().split('\n')
    print(lines)
    pay_on_delivery = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Pay on delivery')), "Yes, this is available {customer_id}")
    returnable = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Returnable')), "Yes, the customer can return fashion.")
    return_window = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Return Window')), "14 days")
    condition_of_items = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "The items must be in their original packaging and unused for a return to be accepted.")
    exceptions = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "Items returned after the 14-day window may not be eligible for a refund, but we will provide a replacement at no additional cost.")
    refunds_exchanges = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "We will process refunds within 3-5 business days of receiving the returned item.")
    additional_notes = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "For Fashion, we recommend checking the product thoroughly upon delivery to ensure it's in right condition before signing for it. If you notice any issues, please contact us immediately.")

    return {"Pay_on_delivery": pay_on_delivery, "Returnable": returnable, "Return_window": return_window, "Condition_of_items": condition_of_items, "Exceptions": exceptions, "Refunds_exchanges": refunds_exchanges, "additional_notes": additional_notes}

@app.post("/medicine")
async def medicine(customer_id: str, customer_score: float):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=700, temperature=0.8, huggingfacehub_api_token=sec_key)

    category_question = """Generate a specific return policy for {category} products for a customer {customer_id} with a trustworthiness score of {customer_score} out of 100. The policy should be lenient for higher scores and stricter for lower scores. Provide the details directly, without headings or redundant information. Limit the response to 3-5 sentences. Ensure the style is consistent across all categories. Format the output as below:
    Pay on delivery: Whether Available [yes] or [no] for that customer and explain about it.
    Returnable: Whether available [yes] or [no] for that customer. If [Yes] then only give the below Return Window for that customer.
    Return Window: Specify the time frame within which returns are accepted for that customer.
    Condition of Items: Mention the condition in which items must be returned for that customer.
    Exceptions and Restrictions: Highlight any exceptions or restrictions that apply for that customer.
    Refunds and Exchanges: State the policy on refunds and exchanges, including who covers return shipping costs if applicable for that customer.
    Additional Notes: Include any additional notes relevant to the return policy for that customer.
    """
    category_policy_prompt_template = PromptTemplate(template=category_question, input_variables=["customer_score", "customer_id", "category"])
    category_policy_chain = LLMChain(llm=llm, prompt = category_policy_prompt_template)

    response = category_policy_chain.invoke({"customer_score": customer_score, "customer_id": customer_id, "category": "medicine"})

    medicine_policy = response['text']
    print(medicine_policy)

    lines = medicine_policy.strip('-').strip().split('\n')
    print(lines)
    pay_on_delivery = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Pay on delivery')), "Yes, this is available for that customer.")
    returnable = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Returnable')), "Yes, the customer can return medicine provided not opening them.")
    return_window = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Return Window')), "14 days")
    condition_of_items = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "The items must be in their original packaging and unused for a return to be accepted.")
    exceptions = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "Items returned after the 14-day window may not be eligible for a refund, but we will provide a replacement at no additional cost.")
    refunds_exchanges = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "We will process refunds within 3-5 business days of receiving the returned item.")
    additional_notes = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "For medicak, we recommend not opening the items if you notice any issues, please contact us immediately.")

    return {"Pay_on_delivery": pay_on_delivery, "Returnable": returnable, "Return_window": return_window, "Condition_of_items": condition_of_items, "Exceptions": exceptions, "Refunds_exchanges": refunds_exchanges, "additional_notes": additional_notes}

@app.post("/beauty")
async def beauty(customer_id: str, customer_score: float):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=700, temperature=0.8, huggingfacehub_api_token=sec_key)

    category_question = """Generate a specific return policy for {category} products for a customer {customer_id} with a trustworthiness score of {customer_score} out of 100. The policy should be lenient for higher scores and stricter for lower scores. Provide the details directly, without headings or redundant information. Limit the response to 3-5 sentences. Ensure the style is consistent across all categories. Format the output as below:
    Pay on delivery: Whether Available [yes] or [no] for that customer and explain about it.
    Returnable: Whether available [yes] or [no] for that customer. If [Yes] then only give the below Return Window for that customer.
    Return Window: Specify the time frame within which returns are accepted for that customer.
    Condition of Items: Mention the condition in which items must be returned for that customer.
    Exceptions and Restrictions: Highlight any exceptions or restrictions that apply for that customer.
    Refunds and Exchanges: State the policy on refunds and exchanges, including who covers return shipping costs if applicable for that customer.
    Additional Notes: Include any additional notes relevant to the return policy for that customer.
    """
    category_policy_prompt_template = PromptTemplate(template=category_question, input_variables=["customer_score", "customer_id", "category"])
    category_policy_chain = LLMChain(llm=llm, prompt = category_policy_prompt_template)

    response = category_policy_chain.invoke({"customer_score": customer_score, "customer_id": customer_id, "category": "beauty and personal"})

    beauty_policy = response['text']
    print(beauty_policy)

    lines = beauty_policy.strip('-').strip().split('\n')
    print(lines)
    pay_on_delivery = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Pay on delivery')), "Yes, this is available for that customer.")
    returnable = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Returnable')), "Yes, the customer can return beauty products provided not using them.")
    return_window = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Return Window')), "14 days")
    condition_of_items = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "The items must be in their original packaging and unused for a return to be accepted.")
    exceptions = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "Items returned after the 14-day window may not be eligible for a refund, but we will provide a replacement at no additional cost.")
    refunds_exchanges = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "We will process refunds within 3-5 business days of receiving the returned item.")
    additional_notes = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "For beauty and personal products, we recommend not opening the items if you notice any issues, please contact us immediately.")

    return {"Pay_on_delivery": pay_on_delivery, "Returnable": returnable, "Return_window": return_window, "Condition_of_items": condition_of_items, "Exceptions": exceptions, "Refunds_exchanges": refunds_exchanges, "additional_notes": additional_notes}

@app.post("/toy")
async def toy(customer_id: str, customer_score: float):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=700, temperature=0.8, huggingfacehub_api_token=sec_key)

    category_question = """Generate a specific return policy for {category} products for a customer {customer_id} with a trustworthiness score of {customer_score} out of 100. The policy should be lenient for higher scores and stricter for lower scores. Provide the details directly, without headings or redundant information. Limit the response to 3-5 sentences. Ensure the style is consistent across all categories. Format the output as below:
    Pay on delivery: Whether Available [yes] or [no] for that customer and explain about it.
    Returnable: Whether available [yes] or [no] for that customer. If [Yes] then only give the below Return Window for that customer.
    Return Window: Specify the time frame within which returns are accepted for that customer.
    Condition of Items: Mention the condition in which items must be returned for that customer.
    Exceptions and Restrictions: Highlight any exceptions or restrictions that apply for that customer.
    Refunds and Exchanges: State the policy on refunds and exchanges, including who covers return shipping costs if applicable for that customer.
    Additional Notes: Include any additional notes relevant to the return policy for that customer.
    """
    category_policy_prompt_template = PromptTemplate(template=category_question, input_variables=["customer_score", "customer_id", "category"])
    category_policy_chain = LLMChain(llm=llm, prompt = category_policy_prompt_template)

    response = category_policy_chain.invoke({"customer_score": customer_score, "customer_id": customer_id, "category": "toys and games"})

    toys_policy = response['text']
    print(toys_policy)

    lines = toys_policy.strip('-').strip().split('\n')
    print(lines)
    pay_on_delivery = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Pay on delivery')), "Yes, this is available for that customer.")
    returnable = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Returnable')), "Yes, the customer can return toys and games provided not using them.")
    return_window = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Return Window')), "14 days")
    condition_of_items = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "The items must be in their original packaging and unused for a return to be accepted.")
    exceptions = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "Items returned after the 14-day window may not be eligible for a refund, but we will provide a replacement at no additional cost.")
    refunds_exchanges = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "We will process refunds within 3-5 business days of receiving the returned item.")
    additional_notes = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "For toys and games, we recommend not opening the items if you notice any issues, please contact us immediately.")

    return {"Pay_on_delivery": pay_on_delivery, "Returnable": returnable, "Return_window": return_window, "Condition_of_items": condition_of_items, "Exceptions": exceptions, "Refunds_exchanges": refunds_exchanges, "additional_notes": additional_notes}

@app.post("/sports")
async def sports(customer_id: str, customer_score: float):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=700, temperature=0.8, huggingfacehub_api_token=sec_key)

    category_question = """Generate a specific return policy for {category} products for a customer {customer_id} with a trustworthiness score of {customer_score} out of 100. The policy should be lenient for higher scores and stricter for lower scores. Provide the details directly, without headings or redundant information. Limit the response to 3-5 sentences. Ensure the style is consistent across all categories. Format the output as below:
    Pay on delivery: Whether Available [yes] or [no] for that customer and explain about it.
    Returnable: Whether available [yes] or [no] for that customer. If [Yes] then only give the below Return Window for that customer.
    Return Window: Specify the time frame within which returns are accepted for that customer.
    Condition of Items: Mention the condition in which items must be returned for that customer.
    Exceptions and Restrictions: Highlight any exceptions or restrictions that apply for that customer.
    Refunds and Exchanges: State the policy on refunds and exchanges, including who covers return shipping costs if applicable for that customer.
    Additional Notes: Include any additional notes relevant to the return policy for that customer.
    """
    category_policy_prompt_template = PromptTemplate(template=category_question, input_variables=["customer_score", "customer_id", "category"])
    category_policy_chain = LLMChain(llm=llm, prompt = category_policy_prompt_template)

    response = category_policy_chain.invoke({"customer_score": customer_score, "customer_id": customer_id, "category": "toys and games"})

    toys_policy = response['text']
    print(toys_policy)

    lines = toys_policy.strip('-').strip().split('\n')
    print(lines)
    pay_on_delivery = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Pay on delivery')), "Yes, this is available for that customer.")
    returnable = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Returnable')), "Yes, the customer can return sports and outdoors provided not using them.")
    return_window = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Return Window')), "14 days")
    condition_of_items = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "The items must be in their original packaging and unused for a return to be accepted.")
    exceptions = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "Items returned after the 14-day window may not be eligible for a refund, but we will provide a replacement at no additional cost.")
    refunds_exchanges = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "We will process refunds within 3-5 business days of receiving the returned item.")
    additional_notes = next((line.split(': ')[1].strip() for line in lines if line.strip().startswith('Condition of Items')), "For sports and outdoors, we recommend not opening the items if you notice any issues, please contact us immediately.")

    return {"Pay_on_delivery": pay_on_delivery, "Returnable": returnable, "Return_window": return_window, "Condition_of_items": condition_of_items, "Exceptions": exceptions, "Refunds_exchanges": refunds_exchanges, "additional_notes": additional_notes}