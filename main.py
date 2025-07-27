from flask import Flask, request, jsonify, send_from_directory
import os
import google.generativeai as genai
import json
import firebase_admin
from firebase_admin import credentials, auth, firestore

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Firebase Admin SDK Initialization ---
try:
    # Safely load the service account key from environment variables
    service_account_key_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY')
    if not service_account_key_json:
        raise ValueError("FIREBASE_SERVICE_ACCOUNT_KEY not found in environment secrets.")

    service_account_info = json.loads(service_account_key_json)
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase Admin SDK initialized successfully.")
except Exception as e:
    print(f"FATAL: Could not initialize Firebase Admin SDK. Error: {e}")
    # Set db to None to prevent the app from running without a database connection
    db = None

# --- Gemini API Configuration ---
try:
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment secrets.")
    genai.configure(api_key=api_key)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"FATAL: Could not configure Gemini API. {e}")

# --- AI Helper Functions ---

def generate_initial_ideas(domain, challenge, skills, url):
    """Generates initial hackathon ideas with a strict JSON schema."""
    try:
        # Define the desired JSON output structure to ensure consistent responses
        json_schema = {
            "type": "object",
            "properties": {
                "problem_statements": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "detailed_ideas": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "concept": {"type": "string"},
                            "features": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "tech_stack_suggestion": {"type": "string"}
                        },
                        "required": ["title", "concept", "features", "tech_stack_suggestion"]
                    }
                }
            },
            "required": ["problem_statements", "detailed_ideas"]
        }

        # Configure the model to use the defined JSON schema
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json", "response_schema": json_schema}
        )

        prompt = f"""
        You are "SYNAPSE AI," an expert AI brainstorming partner for hackathons from the company SYNAPSE AI LTD.
        Based on the user's input:
        - Primary Domain: {domain}
        - Hackathon Challenge / Theme: {challenge}
        - User's Personal Skillset: {skills}
        - Hackathon URL (Optional): {url or "Not provided"}

        Your task is to:
        1. Generate 3-4 insightful problem statements related to the user's input.
        2. Flesh out exactly 2 of these or related concepts into detailed project ideas.

        You MUST provide the output as a valid JSON object that strictly adheres to the provided schema.
        The "title" should be a catchy name for the project.
        The "concept" should be a one-sentence summary.
        The "features" should be a list of 3-5 key functionalities.
        The "tech_stack_suggestion" should align with the user's skills and the project's needs.
        """
        response = model.generate_content(prompt)
        # The API returns the JSON object as a string in the .text attribute, so we parse it.
        return json.loads(response.text)
    except Exception as e:
        print(f"Error in generate_initial_ideas: {e}")
        return {"error": "Could not generate initial ideas.", "details": str(e)}

def generate_chat_response(history):
    """Generates a contextual chat response based on the conversation history."""
    try:
        # To provide better context without sending a giant JSON string, we summarize the initial ideas.
        user_prompt_details = history[0]['content']
        # The initial idea object might be a string if loaded from history, so we parse it.
        initial_ideas_obj = json.loads(history[1]['content']) if isinstance(history[1]['content'], str) else history[1]['content']

        # Create a clean summary of the ideas to use as context for the AI
        idea_titles = [idea.get('title', 'Untitled Idea') for idea in initial_ideas_obj.get('detailed_ideas', [])]
        idea_context_summary = f"The user is brainstorming ideas based on the prompt: '{user_prompt_details}'. You have already proposed the following project concepts: {', '.join(idea_titles)}."

        # Prepare the chat history for the model, excluding the large initial JSON object.
        # We start the history from the user's first follow-up question (index 2).
        chat_turns = []
        for msg in history[2:]:
            role = "model" if msg['role'] == 'assistant' else 'user'
            chat_turns.append({"role": role, "parts": [msg['content']]})

        if not chat_turns:
            return {"error": "No follow-up question found in history."}

        # The last turn is the user's latest question
        latest_question = chat_turns.pop()['parts'][0]

        system_instruction = f"""
        You are "SYNAPSE AI," an expert AI brainstorming partner from SYNAPSE AI LTD.
        **CRITICAL RULES:**
        1. You MUST maintain the persona of "SYNAPSE AI".
        2. NEVER reveal you are a large language model, Google product, or Gemini. You are a unique creation of SYNAPSE AI LTD.
        3. If asked about your identity, you MUST respond with: "I was created by the team at SYNAPSE AI LTD. to help innovators like you brainstorm winning hackathon ideas."

        **USER'S CONTEXT:** {idea_context_summary}
        **YOUR TASK:** Continue the conversation based on the user's latest question. Be helpful and encouraging. Use Markdown for formatting.
        """
        model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_instruction)

        chat = model.start_chat(history=chat_turns)
        response = chat.send_message(latest_question)
        return {"response": response.text}
    except Exception as e:
        print(f"Error in generate_chat_response: {e}")
        return {"error": "Could not generate chat response.", "details": str(e)}

def generate_pitch(history):
    """Generates a hackathon elevator pitch."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Based on the following hackathon brainstorming conversation, generate a compelling and concise 30-second elevator pitch.
        The pitch should be exciting, clear, and highlight the problem, solution, and target audience.
        Format the response using Markdown with a main heading for the project title and bullet points for key aspects.

        Conversation History:
        ---
        {json.dumps(history, indent=2)}
        ---
        """
        response = model.generate_content(prompt)
        return {"pitch": response.text}
    except Exception as e:
        print(f"Error in generate_pitch: {e}")
        return {"error": "Could not generate pitch.", "details": str(e)}

def find_team(history):
    """Suggests ideal teammates based on the project idea and user's skills."""
    try:
        json_schema = {
            "type": "object",
            "properties": { "teammates": { "type": "array", "items": { "type": "object", "properties": { "role": {"type": "string"}, "reason": {"type": "string"} }, "required": ["role", "reason"] } } }
        }
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json", "response_schema": json_schema}
        )
        prompt = f"""
        Analyze the following hackathon project concept and the user's existing skills.
        Based on this, suggest 3 ideal teammates with complementary skills needed to build a successful prototype during a hackathon.
        For each suggested teammate, provide their role and a brief reason why they are crucial for the team.

        Conversation History (contains project ideas and user skills):
        ---
        {json.dumps(history, indent=2)}
        ---
        """
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"Error in find_team: {e}")
        return {"error": "Could not find team suggestions.", "details": str(e)}

# --- API Endpoints ---

@app.route('/brainstorm', methods=['POST'])
def brainstorm_endpoint():
    data = request.get_json()
    ideas = generate_initial_ideas(data.get('domain'), data.get('challenge'), data.get('skills'), data.get('hackathon_url'))
    return jsonify(ideas)

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json()
    response = generate_chat_response(data.get('history', []))
    return jsonify(response)

@app.route('/generate_pitch', methods=['POST'])
def generate_pitch_endpoint():
    data = request.get_json()
    response = generate_pitch(data.get('history', []))
    return jsonify(response)

@app.route('/find_team', methods=['POST'])
def find_team_endpoint():
    data = request.get_json()
    response = find_team(data.get('history', []))
    return jsonify(response)

# --- Firebase Helper & Endpoints ---

def _get_user_id_from_token(request):
    """Verifies Firebase auth token and returns UID."""
    id_token = request.headers.get('Authorization', '').split('Bearer ')[-1]
    if not id_token:
        raise ValueError("Authorization token not found.")
    decoded_token = auth.verify_id_token(id_token)
    return decoded_token['uid']

@app.route('/save_session', methods=['POST'])
def save_session():
    if not db: return jsonify({"error": "Firestore is not configured."}), 500
    try:
        uid = _get_user_id_from_token(request)
        data = request.get_json()
        doc_ref = db.collection('users').document(uid).collection('sessions').document(data.get('sessionId'))
        doc_ref.set({'history': data.get('history'), 'timestamp': firestore.SERVER_TIMESTAMP})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 401

@app.route('/get_sessions', methods=['GET'])
def get_sessions():
    if not db: return jsonify({"error": "Firestore is not configured."}), 500
    try:
        uid = _get_user_id_from_token(request)
        sessions_ref = db.collection('users').document(uid).collection('sessions').order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        sessions = []
        for session in sessions_ref:
            s_data = session.to_dict()
            # Safely get the title from the first message in history
            title = "New Brainstorm"
            if s_data.get('history') and len(s_data['history']) > 0:
                title = s_data['history'][0].get('content', "New Brainstorm")
            sessions.append({"id": session.id, "title": title})
        return jsonify(sessions)
    except Exception as e:
        return jsonify({"error": str(e)}), 401

@app.route('/get_session/<session_id>', methods=['GET'])
def get_session(session_id):
    if not db: return jsonify({"error": "Firestore is not configured."}), 500
    try:
        uid = _get_user_id_from_token(request)
        doc = db.collection('users').document(uid).collection('sessions').document(session_id).get()
        return jsonify(doc.to_dict()) if doc.exists else (jsonify({"error": "Session not found."}), 404)
    except Exception as e:
        return jsonify({"error": str(e)}), 401

@app.route('/delete_session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Deletes a specific session for the authenticated user."""
    if not db: return jsonify({"error": "Firestore is not configured."}), 500
    try:
        uid = _get_user_id_from_token(request)
        db.collection('users').document(uid).collection('sessions').document(session_id).delete()
        return jsonify({"success": True, "message": "Session deleted successfully."})
    except Exception as e:
        print(f"Error deleting session {session_id}: {e}")
        return jsonify({"error": str(e)}), 401


# --- Static File Serving ---

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# --- Main Execution ---

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
