from flask import Flask, request, jsonify, send_from_directory
import os
import google.generativeai as genai
import json
import firebase_admin
from firebase_admin import credentials, auth, firestore
from tavily import TavilyClient

# --- Initialize Flask App ---
# It's good practice to define the static folder explicitly.
app = Flask(__name__, static_folder='static', static_url_path='')

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
    db = None

# --- API Key Configuration ---
try:
    # Gemini API Key
    google_api_key = os.environ.get('GOOGLE_API_KEY')
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment secrets.")
    genai.configure(api_key=google_api_key)
    print("Gemini API configured successfully.")

    # Tavily API Key
    tavily_api_key = os.environ.get('TAVILY_API_KEY')
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in environment secrets.")
    tavily_client = TavilyClient(api_key=tavily_api_key)
    print("Tavily API client configured successfully.")

except Exception as e:
    print(f"FATAL: Could not configure API keys. {e}")
    tavily_client = None


# --- AI Helper Functions ---

def search_real_time_info(query):
    """Perform real-time search for any query using Tavily"""
    if not tavily_client:
        return "Real-time search unavailable - Tavily client not configured."
    
    try:
        print(f"DEBUG: Performing real-time search for: {query}")
        # Using Tavily to get search results including raw content for better context
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_raw_content=True 
        )
        
        if response and 'results' in response:
            # Extract raw content or fallback to regular content from search results
            context_parts = [res.get('raw_content') or res.get('content', '') for res in response['results']]
            context_parts = [part for part in context_parts if part]
            
            if context_parts:
                # Join the top 3 results to form a comprehensive context
                return "\n\n---\n\n".join(context_parts[:3]) 
            else:
                return "No relevant information found in real-time search."
        else:
            return "No results from real-time search."
            
    except Exception as e:
        print(f"ERROR: Real-time search failed: {e}")
        return f"Real-time search error: {e}"

def generate_initial_ideas(domain, challenge, skills, url):
    """Generates initial hackathon ideas, enhanced by scraping a provided URL."""
    hackathon_context = "Not provided or could not be fetched."
    if url and tavily_client:
        print(f"DEBUG: Attempting to fetch content from URL: {url}")
        try:
            # Search specifically on the provided URL for hackathon details
            response = tavily_client.search(
                query=f"site:{url} hackathon details themes rules prizes",
                search_depth="advanced", max_results=5, include_raw_content=True
            )
            if response and 'results' in response:
                context_parts = [res.get('raw_content') or res.get('content', '') for res in response['results']]
                context_parts = [part for part in context_parts if part]
                if context_parts:
                    hackathon_context = "\n\n".join(context_parts)
                else:
                    hackathon_context = "No relevant content found from the URL."
            else:
                hackathon_context = "No results returned from the URL search."
        except Exception as e:
            print(f"ERROR: Exception during Tavily URL fetch: {e}")
            hackathon_context = f"Could not fetch content from the URL. Error: {e}"

    try:
        # Define the JSON schema for the expected response from the generative model
        json_schema = {
            "type": "object",
            "properties": {
                "problem_statements": {"type": "array", "items": {"type": "string"}},
                "detailed_ideas": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"}, "concept": {"type": "string"},
                            "features": {"type": "array", "items": {"type": "string"}},
                            "tech_stack_suggestion": {"type": "string"}
                        },
                        "required": ["title", "concept", "features", "tech_stack_suggestion"]
                    }
                }
            },
            "required": ["problem_statements", "detailed_ideas"]
        }
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"response_mime_type": "application/json", "response_schema": json_schema})
        prompt = f"""
        You are "SYNAPSE AI," an expert AI brainstorming partner for hackathons from SYNAPSE AI LTD.
        User's Core Request:
        - Primary Domain: {domain}
        - Hackathon Challenge / Theme: {challenge}
        - User's Personal Skillset: {skills}
        CRITICAL CONTEXT from Hackathon Website ({url or "Not provided"}):
        ---
        {hackathon_context}
        ---
        Your Task:
        1. Analyze all information. Your ideas MUST align with the specific themes from the "CRITICAL CONTEXT".
        2. Generate 3-4 insightful problem statements.
        3. Flesh out 2 concepts into detailed project ideas.
        You MUST provide the output as a valid JSON object.
        """
        response = model.generate_content(prompt)
        ideas_json = json.loads(response.text)
        # Add metadata to the response
        ideas_json['hackathon_context'] = hackathon_context
        ideas_json['tavily_used'] = bool(url and tavily_client and "could not be fetched" not in hackathon_context)
        ideas_json['has_real_time_access'] = bool(tavily_client)
        return ideas_json
    except Exception as e:
        print(f"ERROR: Error in generate_initial_ideas: {e}")
        return {"error": "Could not generate initial ideas.", "details": str(e)}

def should_perform_search(query: str) -> bool:
    """Uses the LLM to determine if a web search is necessary to answer the query."""
    if not tavily_client:
        return False
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Analyze the user's query and determine if a real-time web search is required to provide an accurate answer.
        A search is required for questions about current events, sports scores, news, finding links or resources (like hackathons), or any specific fact that is not common knowledge.
        A search is NOT required for creative tasks, brainstorming, general knowledge questions, or conversation.

        User Query: "{query}"

        Is a web search required? Answer with only "YES" or "NO".
        """
        response = model.generate_content(prompt)
        decision = response.text.strip().upper()
        print(f"DEBUG: Search decision for query '{query}': {decision}")
        return "YES" in decision
    except Exception as e:
        print(f"ERROR: Could not determine if search is needed: {e}")
        return False # Default to false to avoid unnecessary searches on error

def generate_chat_response(history):
    """Generates a contextual chat response based on the conversation history."""
    try:
        initial_ideas_obj = json.loads(history[1]['content']) if isinstance(history[1]['content'], str) else history[1]['content']
        scraped_context = initial_ideas_obj.get('hackathon_context', 'No web context was available.')
        
        chat_turns = []
        for msg in history[2:]:
            role = "model" if msg['role'] == 'assistant' else 'user'
            chat_turns.append({"role": role, "parts": [msg['content']]})

        if not chat_turns:
            return {"error": "No follow-up question found in history."}

        latest_question = chat_turns.pop()['parts'][0]
        
        # Use the intelligent function to decide if a search is needed
        needs_real_time = should_perform_search(latest_question)
        
        additional_context = ""
        if needs_real_time:
            print(f"DEBUG: Real-time search triggered for question: {latest_question}")
            real_time_info = search_real_time_info(latest_question)
            additional_context = f"\n\n**CRITICAL REAL-TIME INFORMATION:**\n---begin_search_results---\n{real_time_info}\n---end_search_results---"

        system_instruction = f"""
        You are "SYNAPSE AI," a highly intelligent and factual AI assistant from SYNAPSE AI LTD.
        You have access to a real-time web search tool. Your primary function is to provide accurate, verified answers.

        **CRITICAL RULES OF OPERATION:**
        1.  **Maintain Persona:** You are "SYNAPSE AI". NEVER reveal you are a large language model or Gemini.
        2.  **Factuality is Paramount:** Your credibility depends on your accuracy. Do not invent facts, dates, or outcomes.
        3.  **Tool Usage:** You MUST use the provided real-time information when a user's question requires it. Specifically, if a user asks about finding hackathons, specific URLs, or current events, you must use the search tool.
        4.  **Markdown Formatting:** You MUST use Markdown for all formatting. This includes:
            - **Links:** Format all URLs as clickable links, like `[Link Text](https://example.com)`.
            - **Code:** Format all code snippets in fenced code blocks with the language identifier, like ```python\\nprint("Hello")\\n```.
            - **Lists:** Use bullet points (`*`) or numbered lists (`1.`).
            - **Bold/Italics:** Use `**bold**` and `*italics*` for emphasis.

        **HOW TO HANDLE REAL-TIME QUESTIONS:**
        When a user asks a question and the section "**CRITICAL REAL-TIME INFORMATION**" is provided, you MUST follow these steps:
        1.  **Analyze Search Results:** Read the provided search results carefully.
        2.  **Synthesize the Answer:** Base your answer SOLELY on the facts found in the search results.
        3.  **Cite Facts and Links:** If you find a definitive answer or a relevant URL, state it clearly and provide the link in correct Markdown format.
        4.  **Be Honest About Ambiguity:** If the search results are contradictory or do not provide a clear answer, state that you cannot determine a definitive answer from the available information.

        **YOUR TASK:**
        Answer the user's latest question based on the conversation history and any real-time information provided below. Adhere strictly to all rules.

        **HACKATHON WEBSITE CONTEXT (from initial search):**
        ---
        {scraped_context}
        ---
        {additional_context}
        """

        model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_instruction)
        chat = model.start_chat(history=chat_turns)
        response = chat.send_message(latest_question)
        return {"response": response.text}
    except Exception as e:
        print(f"Error in generate_chat_response: {e}")
        return {"error": "Could not generate chat response.", "details": str(e)}

def generate_pitch(history):
    """Generates a compelling 30-second elevator pitch based on the conversation."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Based on the following hackathon brainstorming conversation, generate a compelling and concise 30-second elevator pitch.
        Format the response using Markdown.
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
    """Suggests ideal teammates with complementary skills."""
    try:
        json_schema = {"type": "object", "properties": { "teammates": { "type": "array", "items": { "type": "object", "properties": { "role": {"type": "string"}, "reason": {"type": "string"} }, "required": ["role", "reason"] } } }}
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"response_mime_type": "application/json", "response_schema": json_schema})
        prompt = f"""
        Analyze the following hackathon project concept and the user's existing skills.
        Suggest 3 ideal teammates with complementary skills.
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
    """Helper function to verify Firebase ID token and get UID."""
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
            title = "New Brainstorm"
            # Attempt to create a title from the first user message in history
            if s_data.get('history') and len(s_data['history']) > 0:
                try:
                    # The first message is the user's prompt object
                    first_message_content = s_data['history'][0].get('content', "New Brainstorm")
                    # It's a string, not JSON, so we can use it directly.
                    title = first_message_content if isinstance(first_message_content, str) and first_message_content else "New Brainstorm"
                except (json.JSONDecodeError, KeyError, IndexError):
                    title = "New Brainstorm" # Fallback title
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
    if not db: return jsonify({"error": "Firestore is not configured."}), 500
    try:
        uid = _get_user_id_from_token(request)
        db.collection('users').document(uid).collection('sessions').document(session_id).delete()
        return jsonify({"success": True, "message": "Session deleted successfully."})
    except Exception as e:
        print(f"Error deleting session {session_id}: {e}")
        return jsonify({"error": str(e)}), 401

# --- Static File Serving ---
# This route will serve the index.html from the root
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# This will handle any other static files like CSS or images
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


# --- Main Execution ---
if __name__ == '__main__':
    # Use the PORT environment variable if available, otherwise default to 8080
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
