from flask import Flask, request, jsonify, render_template
import os
from datetime import datetime
import requests
import json
import re
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Configure Hugging Face API
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize scheduler for reminders
scheduler = BackgroundScheduler()
scheduler.start()
reminders = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_request():
    data = request.json
    user_input = data.get('message', '')
    
    # Call the Hugging Face AI
    response = get_ai_response(user_input)
    
    # Extract the clean JSON response from Hugging Face
    action = parse_action(response)
    
    # Convert JSON to human-readable response
    human_response = generate_human_response(action)
    
    return jsonify(human_response)

# Function to Call Hugging Face AI
def get_ai_response(user_input):
    prompt = f"""
<s>[INST]
You are a helpful virtual assistant that can:
- Answer general questions in natural language.
- Set reminders, send emails, or perform web searches in JSON format.

IMPORTANT RULES:
- If the user asks general questions, answer them conversationally.
- If the user gives a task (reminder, email, web search), respond in JSON.
- NEVER use history or previous conversation.
- DO NOT include the user's input in your response.

User request: {user_input}
[/INST]</s>
    """
    
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json={"inputs": prompt})
        print(response.json())  # Debug print
        
        if response.status_code == 503:
            return json.dumps({"action": "chat", "details": {"message": "I'm still loading. Please try again later."}})
        
        response_text = response.json()[0].get("generated_text", "")
        
        # Look for JSON in response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                json_obj = json.loads(json_str)
                # If it has an answer key, treat it as a chat response
                if "answer" in json_obj:
                    return json.dumps({
                        "action": "chat",
                        "details": {"message": json_obj["answer"]}
                    })
                else:
                    return json_str  # Return the JSON as is for actions
            except json.JSONDecodeError:
                return force_chat_response(extract_answer(response_text))
        else:
            return force_chat_response(extract_answer(response_text))
    
    except Exception as e:
        print(f"Error calling Hugging Face API: {e}")
        return force_default_response(user_input)

# Extract the natural language response
def extract_answer(response_text):
    answer_match = re.search(r'"answer":\s*"(.*?)"', response_text)
    if answer_match:
        return answer_match.group(1)
    else:
        return response_text

# Force normal chat response
def force_chat_response(response_text):
    return json.dumps({
        "action": "chat",
        "details": {"message": response_text}
    })

# Validate JSON response
def is_valid_json(response):
    try:
        json.loads(response)
        return True
    except ValueError:
        return False

# Force JSON if response fails
def force_default_response(user_input):
    return json.dumps({
        "action": "chat",
        "details": {"message": f"I'm not sure about '{user_input}', but I can help you with something else."}
    })

# Handle web search, email, and reminders
def parse_action(ai_response):
    try:
        parsed = json.loads(ai_response)
        return parsed
    except json.JSONDecodeError:
        return {"action": "chat", "details": {"message": ai_response}}

# Convert JSON to Human-readable response
def generate_human_response(action):
    action_type = action.get('action', 'chat')
    details = action.get('details', {})

    if action_type == 'reminder':
        return {
            'response': f"✅ Reminder Set: '{details.get('message')}' at {details.get('time')}",
            'type': 'reminder'
        }
    elif action_type == 'email':
        return {
            'response': f"📧 Email Prepared: To {details.get('to')} with subject '{details.get('subject')}'",
            'type': 'email'
        }
    elif action_type == 'websearch':
        return {
            'response': f"🌐 Searching Google for: '{details.get('query')}'",
            'type': 'websearch'
        }
    else:
        # Fix newline formatting in chat responses
        message = details.get('message', '')
        # Replace literal \n with actual newlines if they exist
        message = message.replace('\\n', '\n')
        return {
            'response': message,
            'type': 'chat'
        }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)