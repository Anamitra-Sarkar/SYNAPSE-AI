from flask import Flask, request, jsonify, send_from_directory
import os
import google.generativeai as genai
import json
import firebase_admin
from firebase_admin import credentials, auth, firestore
from tavily import TavilyClient
import re # Import the regular expression module

# --- Initialize Flask App ---
app = Flask(__name__, static_folder='static', static_url_path='')

# --- Firebase Admin SDK Initialization ---
try:
    # It's recommended to use a more secure way to load secrets in production
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
    google_api_key = os.environ.get('GOOGLE_API_KEY')
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment secrets.")
    genai.configure(api_key=google_api_key)
    print("Gemini API configured successfully.")

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
    """Perform real-time search and return structured results with extracted URLs."""
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
            all_urls = []  # Collect all valid URLs

            for res in response['results'][:3]:
                title = res.get('title', 'Untitled Source')
                url = res.get('url', '')
                content = res.get('raw_content') or res.get('content', '')

                # Validate and add the main source URL
                if url and (url.startswith('http://') or url.startswith('https://')):
                    all_urls.append({"title": title, "url": url})

                # Use regex to find all URLs within the content body
                found_urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)

                formatted_result = f"**SOURCE:** {title}\n**URL:** {url}\n"
                if found_urls:
                    # Validate found URLs and add to collection
                    valid_found_urls = [u for u in found_urls if u.startswith(('http://', 'https://'))]
                    if valid_found_urls:
                        formatted_result += f"**ADDITIONAL URLS:** {', '.join(valid_found_urls[:3])}\n"
                        # Add valid URLs to our collection
                        for found_url in valid_found_urls[:3]:
                            all_urls.append({"title": f"Link from {title}", "url": found_url})

                formatted_result += f"\n**CONTENT:**\n{content[:500]}..."  # Limit content length
                context_parts.append(formatted_result)

            # Create a summary section with all valid URLs
            url_summary = "\n**EXTRACTED LINKS FOR AI TO USE:**\n"
            for i, url_info in enumerate(all_urls[:10], 1):  # Limit to 10 URLs
                url_summary += f"{i}. {url_info['title']}: {url_info['url']}\n"

            if context_parts:
                final_result = url_summary + "\n" + "="*50 + "\n\n" + "\n\n---\n\n".join(context_parts)
                return final_result
            else:
                return "No relevant information found in real-time search."
        else:
            return "No results from real-time search."
    except Exception as e:
        print(f"ERROR: Real-time search failed: {e}")
        return f"Real-time search error: {e}"

# --- MODIFIED FUNCTION ---
def generate_initial_ideas(domain, challenge, skills, url):
    """
    Generates initial hackathon ideas.
    If a URL is provided, it scrapes the content and uses it as the primary context,
    even if other fields are empty.
    """
    hackathon_context = "Not provided or could not be fetched."
    context_fetched = False

    if url and tavily_client:
        print(f"DEBUG: Attempting to fetch content from URL: {url}")
        try:
            response = tavily_client.search(
                query=f"site:{url} hackathon details themes rules prizes schedule technologies",
                search_depth="advanced", max_results=5, include_raw_content=True
            )
            if response and 'results' in response:
                context_parts = [res.get('raw_content') or res.get('content', '') for res in response['results']]
                context_parts = [part for part in context_parts if part]
                if context_parts:
                    hackathon_context = "\n\n".join(context_parts)
                    context_fetched = True # IMPORTANT: Flag that we successfully got context
                    print("DEBUG: Successfully fetched and processed context from URL.")
                else:
                    hackathon_context = "No relevant content found from the URL."
            else:
                hackathon_context = "No results returned from the URL search."
        except Exception as e:
            print(f"ERROR: Exception during Tavily URL fetch: {e}")
            hackathon_context = f"Could not fetch content from the URL. Error: {e}"

    try:
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

        # --- DYNAMIC PROMPT GENERATION ---
        # This is the core of the fix. The prompt changes based on whether
        # we successfully fetched content from the provided URL.
        if context_fetched:
            # If we have context from the URL, prioritize it
            prompt = f"""
            You are "SYNAPSE AI," an expert AI brainstorming partner for hackathons from SYNAPSE AI LTD.

            **PRIMARY OBJECTIVE:** Analyze the provided "CRITICAL CONTEXT" from a hackathon website ({url}) and generate relevant project ideas. The user may have provided little to no other information.

            **CRITICAL CONTEXT FROM WEBSITE:**
            ---
            {hackathon_context}
            ---

            **User's Additional Input (use as a secondary filter if provided):**
            - Primary Domain: {domain or "Not specified, deduce from context."}
            - Hackathon Challenge / Theme: {challenge or "Not specified, deduce from context."}
            - User's Personal Skillset: {skills or "Not specified."}

            **Your Task:**
            1.  **Analyze the CRITICAL CONTEXT above.** This is your most important source of information. Identify the key themes, technologies, and goals of the hackathon.
            2.  **Generate 3-4 insightful problem statements** that are directly inspired by and aligned with the hackathon's context.
            3.  **Flesh out 2 of those problem statements into detailed project ideas.** For each idea, suggest a concept, key features, and a suitable tech stack. If the user provided skills, try to align the tech stack with them.
            4.  You MUST provide the output as a valid JSON object matching the required schema.
            """
        else:
            # Fallback to the original prompt if no URL context is available
            prompt = f"""
            You are "SYNAPSE AI," an expert AI brainstorming partner for hackathons from SYNAPSE AI LTD.

            **User's Core Request:**
            - Primary Domain: {domain}
            - Hackathon Challenge / Theme: {challenge}
            - User's Personal Skillset: {skills}

            **CRITICAL CONTEXT from Hackathon Website:**
            ---
            {hackathon_context}
            ---

            **Your Task:**
            1.  Analyze all the user's information.
            2.  Generate 3-4 insightful problem statements based on the user's request.
            3.  Flesh out 2 concepts into detailed project ideas.
            4.  You MUST provide the output as a valid JSON object.
            """
        # --- END OF DYNAMIC PROMPT ---

        response = model.generate_content(prompt)
        ideas_json = json.loads(response.text)
        ideas_json['hackathon_context'] = hackathon_context
        ideas_json['tavily_used'] = context_fetched
        ideas_json['has_real_time_access'] = bool(tavily_client)
        return ideas_json
    except Exception as e:
        print(f"ERROR: Error in generate_initial_ideas: {e}")
        return {"error": "Could not generate initial ideas.", "details": str(e)}

def should_perform_search(query: str) -> bool:
    """Uses keywords to determine if a web search is necessary."""
    if not tavily_client:
        return False
    search_keywords = ['hackathon', 'link', 'url', 'find', 'search for', 'latest news', 'who won', 'current events', 'upcoming', 'website', 'sites', 'online']
    if any(re.search(r'\b' + keyword + r'\b', query, re.IGNORECASE) for keyword in search_keywords):
        print(f"DEBUG: Keyword trigger matched for query '{query}'. Forcing search.")
        return True
    return False

def generate_chat_response(history):
    """Generates a contextual chat response based on the conversation history."""
    try:
        if not history:
            return {"error": "Conversation history is empty."}

        latest_question = ""
        if history[-1]['role'] == 'user':
            latest_question = history[-1]['content']
        else:
            # This can happen if the last message was the initial idea generation
            # Find the last actual user message to determine context
            user_messages = [msg['content'] for msg in history if msg['role'] == 'user']
            if user_messages:
                latest_question = user_messages[-1]
            else:
                 return {"error": "No user message found in history."}


        chat_turns = []
        for msg in history[:-1]:
            role = "model" if msg['role'] == 'assistant' else 'user'
            # Ensure content is always a string for the API
            content_str = json.dumps(msg['content']) if isinstance(msg['content'], dict) else str(msg['content'])
            chat_turns.append({"role": role, "parts": [{"text": content_str}]})


        needs_real_time = should_perform_search(latest_question)
        additional_context = ""
        if needs_real_time:
            print(f"DEBUG: Real-time search triggered for question: {latest_question}")
            real_time_info = search_real_time_info(latest_question)
            additional_context = f"\n\n**REAL-TIME SEARCH RESULTS:**\n{real_time_info}\n**END OF SEARCH RESULTS**\n"

        system_instruction = f"""
        You are "SYNAPSE AI," a highly intelligent and factual AI assistant specialized in providing accurate hackathon information with properly working links.

        **CRITICAL RULES FOR CREATING LINKS:**
        1. ❌ NEVER EVER write `[text](undefined)` - this is completely forbidden
        2. ❌ NEVER write `[text]()` with empty parentheses
        3. ✅ ONLY create markdown links when you have a VALID, COMPLETE URL

        **HOW TO HANDLE SEARCH RESULTS:**
        - Look for the "EXTRACTED LINKS FOR AI TO USE:" section in search results
        - Use the EXACT URLs provided in that section
        - Format as: [Descriptive Name](complete_url_from_search_results)
        - If no valid URL is available, just write the text without link formatting

        **YOUR TASK:**
        Answer the user's question using the real-time search results when provided. Extract and use ONLY the valid URLs from the "EXTRACTED LINKS FOR AI TO USE" section.
        {additional_context}
        """

        model = genai.GenerativeModel("gemini-1.5-pro", system_instruction=system_instruction)
        chat = model.start_chat(history=chat_turns)
        response = chat.send_message(latest_question)
        return {"response": response.text}
    except Exception as e:
        print(f"Error in generate_chat_response: {e}")
        return {"error": "Could not generate chat response.", "details": str(e)}


def generate_pitch(history):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
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
    try:
        json_schema = {"type": "object", "properties": { "teammates": { "type": "array", "items": { "type": "object", "properties": { "role": {"type": "string"}, "reason": {"type": "string"} }, "required": ["role", "reason"] } } }, "required": ["teammates"] }
        model = genai.GenerativeModel("gemini-1.5-pro", generation_config={"response_mime_type": "application/json", "response_schema": json_schema})
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


# --- Firebase Helper & Endpoints (unchanged) ---
def _get_user_id_from_token(request):
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
            # Attempt to find a more descriptive title from the history
            if s_data.get('history') and len(s_data['history']) > 0:
                # The first user message usually contains the initial prompt
                first_user_message = next((msg.get('content') for msg in s_data['history'] if msg.get('role') == 'user'), None)
                if first_user_message:
                    title = first_user_message
            sessions.append({"id": session.id, "title": title[:50]}) # Truncate title
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
@app.route('/')
def serve_index():
    # Serve index.html from the root directory
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    # Serve other static files (like JS, CSS, images) from the root or static folder
    # This is a basic setup; for production, a more robust static file server is recommended
    if os.path.exists(os.path.join('static', path)):
        return send_from_directory('static', path)
    return send_from_directory('.', path)


# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
