from flask import Flask, request, jsonify, render_template
import os
from datetime import datetime
import requests
import json
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Configure Hugging Face API
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Email configuration
EMAIL_HOST = os.getenv('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', 587))
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
EMAIL_FROM = os.getenv('EMAIL_FROM', EMAIL_USER)

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
    
    # Execute the action if needed
    result = execute_action(action)
    
    # Convert JSON to human-readable response
    human_response = generate_human_response(action, result)
    
    return jsonify(human_response)

# Function to Call Hugging Face AI with improved prompt
# Replace your get_ai_response function with this improved version
def get_ai_response(user_input):
    prompt = f"""
<s>[INST]
You are a helpful virtual assistant that can:
- Answer general questions in natural language.
- Set reminders, send emails, or perform web searches.

For tasks, respond with structured JSON in this format:
1. For emails:
```json
{{
  "action": "email",
  "details": {{
    "to": "recipient@example.com",
    "subject": "Email subject",
    "body": "Full email body with detailed content",
    "cc": "optional_cc@example.com"
  }}
}}
```

2. For reminders:
```json
{{
  "action": "reminder",
  "details": {{
    "time": "YYYY-MM-DD HH:MM",
    "message": "Reminder message"
  }}
}}
```

3. For web searches:
```json
{{
  "action": "websearch",
  "details": {{
    "query": "Search query text"
  }}
}}
```

4. For normal conversation or answering questions, always use this format:
```json
{{
  "action": "chat",
  "details": {{
    "message": "Your complete answer here"
  }}
}}
```

IMPORTANT:
- Always include complete information in JSON responses.
- For emails, provide a detailed body with proper formatting.
- Use the exact JSON structure shown above.
- DO NOT include the user's input in your response.
- For questions about facts, information, or conversation, ALWAYS use the chat format.

User request: {user_input}
[/INST]</s>
    """
    
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json={"inputs": prompt})
        
        if response.status_code == 503:
            return json.dumps({"action": "chat", "details": {"message": "I'm still loading. Please try again later."}})
        
        # Get the full response text
        response_text = response.json()[0].get("generated_text", "")
        
        # Extract only the response part after the instruction
        clean_response_match = re.search(r'\[\/INST\]\<\/s\>(.*)', response_text, re.DOTALL)
        
        if clean_response_match:
            clean_response = clean_response_match.group(1).strip()
            
            # Look for JSON pattern in the response
            json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    json_obj = json.loads(json_str)
                    return json.dumps(json_obj)
                except json.JSONDecodeError:
                    # If JSON parsing fails, treat as regular chat
                    return force_chat_response(clean_response)
            else:
                # No JSON found, treat as regular chat
                return force_chat_response(clean_response)
        else:
            # Fallback if we can't extract cleanly
            return force_chat_response(response_text)
    
    except Exception as e:
        print(f"Error calling Hugging Face API: {e}")
        return force_default_response(user_input)

# Improved version of force_chat_response
def force_chat_response(response_text):
    # Clean up the response text if needed
    cleaned_text = response_text.strip()
    
    # If it's very short or empty, provide a fallback
    if len(cleaned_text) < 5:
        cleaned_text = "I'm sorry, I don't have a good answer for that."
    
    return json.dumps({
        "action": "chat",
        "details": {"message": cleaned_text}
    })

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

# Fixed function to execute actions
def execute_action(action):
    action_type = action.get('action', 'chat')
    details = action.get('details', {})
    result = {"success": True, "message": ""}
    
    try:
        if action_type == 'email':
            # Actually send the email
            if not EMAIL_USER or not EMAIL_PASSWORD:
                result = {"success": False, "message": "Email credentials not configured"}
            else:
                to_email = details.get('to', '')
                subject = details.get('subject', 'No Subject')
                body = details.get('body', '')
                cc = details.get('cc', None)
                
                # Send the email
                send_email(to_email, subject, body, cc)
                result = {"success": True, "message": "Email sent successfully"}
        
        elif action_type == 'reminder':
            # Fixed reminder handling
            time_str = details.get('time', '')
            message = details.get('message', '')
            
            # Validate time format
            try:
                reminder_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                # Add reminder to list
                reminder_id = len(reminders)
                reminder = {
                    "id": reminder_id,
                    "time": time_str,
                    "message": message
                }
                reminders.append(reminder)
                
                # Schedule the reminder notification (implementation would depend on your needs)
                result = {"success": True, "message": "Reminder set successfully"}
            except ValueError:
                # Handle incorrect time format
                result = {"success": False, "message": "Invalid time format. Please use YYYY-MM-DD HH:MM"}
            
        elif action_type == 'websearch':
            query = details.get('query', '')
            # Here you would typically implement actual search functionality
            # For now, we're just acknowledging the search request
            result = {"success": True, "message": f"Search initiated for: {query}"}
            
    except Exception as e:
        result = {"success": False, "message": str(e)}
        
    return result

# Function to send emails
def send_email(to_email, subject, body, cc=None):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = to_email
        msg['Subject'] = subject
        
        if cc:
            msg['Cc'] = cc
            
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        
        recipients = [to_email]
        if cc:
            recipients.append(cc)
            
        server.sendmail(EMAIL_FROM, recipients, msg.as_string())
        server.quit()
        
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        raise

# Convert JSON to Human-readable response with results
# Replace this section in your generate_human_response function
def generate_human_response(action, result=None):
    action_type = action.get('action', 'chat')
    details = action.get('details', {})
    
    if action_type == 'email':
        # Email handling (keep your existing code)
        response = f"📧 Email to {details.get('to')}\n"
        response += f"Subject: {details.get('subject')}\n\n"
        
        # Include a preview of the email body (first 100 chars)
        body = details.get('body', '')
        body_preview = body[:100] + ('...' if len(body) > 100 else '')
        response += f"Body: {body_preview}\n\n"
        
        # Add status message
        if result and not result.get('success', True):
            response += f"⚠️ {result.get('message', 'Failed to send email')}"
        else:
            response += "✅ Email prepared successfully"
            
        return {
            'response': response,
            'type': 'email',
            'details': details  # Pass through full details for potential UI use
        }
        
    elif action_type == 'reminder':
        # Reminder handling (keep your existing code)
        time_str = details.get('time', 'Not specified')
        message = details.get('message', 'No message')
        
        if result and not result.get('success', True):
            return {
                'response': f"⚠️ Failed to set reminder: {result.get('message')}",
                'type': 'reminder',
                'details': details
            }
        else:
            return {
                'response': f"⏰ Reminder Set: '{message}' at {time_str}",
                'type': 'reminder',
                'details': details
            }
        
    elif action_type == 'websearch':
        # Web search handling (keep your existing code)
        query = details.get('query', 'Not specified')
        return {
            'response': f"🔍 Searching for: '{query}'",
            'type': 'websearch',
            'details': details
        }
        
    else:
        # Improved chat response handling
        message = details.get('message', '')
        if not message:
            message = "I'm sorry, I don't have a good answer for that."
            
        # Ensure literal newlines are converted to actual newlines
        message = message.replace('\\n', '\n')
        
        return {
            'response': message,
            'type': 'chat'
        }

# Route to view/manage reminders
@app.route('/api/reminders', methods=['GET'])
def get_reminders():
    return jsonify(reminders)

# Add this route to your Flask app for handling email sending directly
@app.route('/api/send-email', methods=['POST'])
def send_email_route():
    try:
        data = request.json
        to_email = data.get('to', '')
        subject = data.get('subject', 'No Subject')
        body = data.get('body', '')
        cc = data.get('cc', None)
        
        if not EMAIL_USER or not EMAIL_PASSWORD:
            return jsonify({"success": False, "message": "Email credentials not configured"})
        
        send_email(to_email, subject, body, cc)
        return jsonify({"success": True, "message": "Email sent successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)