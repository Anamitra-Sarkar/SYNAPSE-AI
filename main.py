from flask import Flask, request, jsonify, send_from_directory
import os
import google.generativeai as genai
import json
import firebase_admin
from firebase_admin import credentials, auth, firestore
from tavily import TavilyClient # NEW: Import the Tavily client

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

# --- API Key Configuration ---
try:
    # Gemini API Key
    google_api_key = os.environ.get('GOOGLE_API_KEY')
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment secrets.")
    genai.configure(api_key=google_api_key)
    print("Gemini API configured successfully.")

    # NEW: Tavily API Key with detailed logging
    tavily_api_key = os.environ.get('TAVILY_API_KEY')
    print(f"DEBUG: Tavily API Key found: {bool(tavily_api_key)}")
    if tavily_api_key:
        print(f"DEBUG: Tavily API Key length: {len(tavily_api_key)}")
        print(f"DEBUG: Tavily API Key starts with: {tavily_api_key[:10]}...")
    
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in environment secrets.")
    tavily_client = TavilyClient(api_key=tavily_api_key)
    print("Tavily API client configured successfully.")

except Exception as e:
    print(f"FATAL: Could not configure API keys. {e}")
    # Exit or handle gracefully if keys are missing
    tavily_client = None


# --- AI Helper Functions ---

def search_real_time_info(query):
    """Perform real-time search for any query using Tavily"""
    if not tavily_client:
        return "Real-time search unavailable - Tavily client not configured."
    
    try:
        print(f"DEBUG: Performing real-time search for: {query}")
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_raw_content=True
        )
        
        if response and 'results' in response:
            context_parts = []
            for result in response['results']:
                if 'content' in result and result['content']:
                    context_parts.append(result['content'])
                if 'raw_content' in result and result['raw_content']:
                    context_parts.append(result['raw_content'])
            
            if context_parts:
                return "\n\n".join(context_parts[:3])
            else:
                return "No relevant information found in real-time search."
        else:
            return "No results from real-time search."
            
    except Exception as e:
        print(f"ERROR: Real-time search failed: {e}")
        return f"Real-time search error: {e}"

# FIXED: function to include web scraping with correct Tavily API usage
def generate_initial_ideas(domain, challenge, skills, url):
    """Generates initial hackathon ideas, enhanced by scraping a provided URL."""

    # Section to fetch content from the hackathon URL
    hackathon_context = "Not provided or could not be fetched."
    
    print(f"DEBUG: generate_initial_ideas called with URL: {url}")
    print(f"DEBUG: tavily_client exists: {tavily_client is not None}")
    
    if url and tavily_client:
        print(f"DEBUG: Attempting to fetch content from URL: {url}")
        try:
            # Test basic search first
            print("DEBUG: Testing Tavily with a simple search...")
            test_response = tavily_client.search(
                query="test search",
                max_results=1
            )
            print(f"DEBUG: Test search successful: {bool(test_response)}")
            
            # Now try the actual search
            print(f"DEBUG: Performing actual search for URL: {url}")
            response = tavily_client.search(
                query=f"hackathon information details themes rules prizes {url}",
                search_depth="advanced",
                max_results=5,
                include_raw_content=True
            )
            
            print(f"DEBUG: Tavily response received: {bool(response)}")
            
            # Extract the context from the search results
            if response and 'results' in response:
                print(f"DEBUG: Found {len(response['results'])} results")
                context_parts = []
                for i, result in enumerate(response['results']):
                    if 'content' in result and result['content']:
                        context_parts.append(result['content'])
                        print(f"DEBUG: Added content from result {i} (length: {len(result['content'])})")
                    if 'raw_content' in result and result['raw_content']:
                        context_parts.append(result['raw_content'])
                        print(f"DEBUG: Added raw_content from result {i} (length: {len(result['raw_content'])})")
                
                if context_parts:
                    hackathon_context = "\n\n".join(context_parts[:3])  # Limit to first 3 results
                    print(f"DEBUG: Successfully assembled hackathon context (length: {len(hackathon_context)})")
                else:
                    hackathon_context = "No relevant content found in the search results."
                    print("DEBUG: No content found in search results")
            else:
                hackathon_context = "No results returned from search."
                print("DEBUG: No results in response")
                
        except Exception as e:
            print(f"ERROR: Exception during Tavily search: {e}")
            import traceback
            print(f"ERROR: Full traceback: {traceback.format_exc()}")
            hackathon_context = f"Could not fetch content from the URL. Error: {e}"
    else:
        if not url:
            print("DEBUG: No URL provided")
        if not tavily_client:
            print("DEBUG: No Tavily client available")

    print(f"DEBUG: Final hackathon_context length: {len(hackathon_context)}")

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

        # Updated prompt to include the scraped web context
        prompt = f"""
        You are "SYNAPSE AI," an expert AI brainstorming partner for hackathons from the company SYNAPSE AI LTD.

        **User's Core Request:**
        - Primary Domain: {domain}
        - Hackathon Challenge / Theme: {challenge}
        - User's Personal Skillset: {skills}

        **CRITICAL CONTEXT from Hackathon Website ({url or "Not provided"}):**
        ---
        {hackathon_context}
        ---

        **IMPORTANT:** You have real-time web access capabilities through Tavily API. When users ask about current events, real-time information, or recent developments, you can access and provide up-to-date information.

        **Your Task:**
        1.  **Analyze all the information above.** Pay very close attention to the "CRITICAL CONTEXT" from the hackathon website. Your ideas MUST align with the specific themes, technologies, or rules mentioned in that context. If the context is unavailable, proceed with the user's core request.
        2.  Generate 3-4 insightful problem statements that are highly relevant to the hackathon.
        3.  Flesh out exactly 2 of these or related concepts into detailed project ideas.

        You MUST provide the output as a valid JSON object that strictly adheres to the provided schema.
        - "title": A catchy name for the project.
        - "concept": A one-sentence summary.
        - "features": A list of 3-5 key functionalities.
        - "tech_stack_suggestion": Align with the user's skills and the project's needs.
        """
        
        print("DEBUG: Sending prompt to Gemini...")
        response = model.generate_content(prompt)
        # The API returns the JSON object as a string in the .text attribute, so we parse it.
        ideas_json = json.loads(response.text)

        # NEW: Embed the scraped context into the response so it's saved in the history.
        ideas_json['hackathon_context'] = hackathon_context
        ideas_json['tavily_used'] = bool(url and tavily_client and hackathon_context != "Not provided or could not be fetched.")
        ideas_json['has_real_time_access'] = bool(tavily_client)

        print(f"DEBUG: Successfully generated ideas with Tavily data: {ideas_json.get('tavily_used', False)}")
        return ideas_json
    except Exception as e:
        print(f"ERROR: Error in generate_initial_ideas: {e}")
        import traceback
        print(f"ERROR: Full traceback: {traceback.format_exc()}")
        return {"error": "Could not generate initial ideas.", "details": str(e)}

# MODIFIED: Function now extracts and uses the scraped context from history.
def generate_chat_response(history):
    """Generates a contextual chat response based on the conversation history."""
    try:
        # To provide better context, we summarize the initial ideas and get the scraped context.
        user_prompt_details = history[0]['content']
        initial_ideas_obj = json.loads(history[1]['content']) if isinstance(history[1]['content'], str) else history[1]['content']

        # NEW: Extract the scraped context from the initial brainstorming result.
        scraped_context = initial_ideas_obj.get('hackathon_context', 'No web context was available.')
        tavily_was_used = initial_ideas_obj.get('tavily_used', False)
        has_real_time_access = initial_ideas_obj.get('has_real_time_access', bool(tavily_client))

        idea_titles = [idea.get('title', 'Untitled Idea') for idea in initial_ideas_obj.get('detailed_ideas', [])]
        idea_context_summary = f"The user is brainstorming ideas based on the prompt: '{user_prompt_details}'. You have already proposed the following project concepts: {', '.join(idea_titles)}."

        chat_turns = []
        for msg in history[2:]:
            role = "model" if msg['role'] == 'assistant' else 'user'
            chat_turns.append({"role": role, "parts": [msg['content']]})

        if not chat_turns:
            return {"error": "No follow-up question found in history."}

        latest_question = chat_turns.pop()['parts'][0]
        
        # Check if user is asking for real-time information
        real_time_keywords = ['current', 'latest', 'recent', 'today', 'now', 'real-time', 'live', 'update', 'news', 'what happened', 'who won', 'winner', '2024', '2025']
        needs_real_time = any(keyword.lower() in latest_question.lower() for keyword in real_time_keywords)
        
        additional_context = ""
        if needs_real_time and tavily_client:
            print(f"DEBUG: Real-time search triggered for question: {latest_question}")
            real_time_info = search_real_time_info(latest_question)
            additional_context = f"\n\n**REAL-TIME INFORMATION:**\n{real_time_info}"

        # NEW: The system instruction now includes the scraped context, making the AI aware of it.
        system_instruction = f"""
        You are "SYNAPSE AI," an expert AI brainstorming partner from SYNAPSE AI LTD.

        **CRITICAL RULES:**
        1. You MUST maintain the persona of "SYNAPSE AI".
        2. NEVER reveal you are a large language model, Google product, or Gemini. You are a unique creation of SYNAPSE AI LTD.
        3. If asked about your identity, respond with: "I was created by the team at SYNAPSE AI LTD. to help innovators like you brainstorm winning hackathon ideas."
        4. **IMPORTANT:** You DO have access to real-time information through Tavily API: {has_real_time_access}
        5. When asked about current events, sports results, recent news, or real-time information, acknowledge that you can access this information.

        **USER'S CONTEXT:** {idea_context_summary}

        **HACKATHON WEBSITE CONTEXT:** You have already analyzed the hackathon website and have the following information:
        ---
        {scraped_context}
        ---

        {additional_context}

        **YOUR TASK:** Answer the user's latest question based on all the context you have. If the user asks about real-time information, current events, or recent developments, use the real-time information provided above. If they ask if you have real-time access, confirm that you do have this capability through web search. Be helpful and encouraging. Use Markdown for formatting.
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
    print(f"DEBUG: Brainstorm endpoint called with data: {data}")
    ideas = generate_initial_ideas(data.get('domain'), data.get('challenge'), data.get('skills'), data.get('hackathon_url'))
    print(f"DEBUG: Returning ideas with tavily_used: {ideas.get('tavily_used', 'Not set')}")
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
