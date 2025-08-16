from flask import Flask, request, jsonify, send_from_directory
import os
import re
import json
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, auth, firestore
from tavily import TavilyClient

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
    db = None

# --- API Key Configuration ---
google_configured = False
try:
    # Gemini API Key
    google_api_key = os.environ.get('GOOGLE_API_KEY')
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment secrets.")
    genai.configure(api_key=google_api_key)
    google_configured = True
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"FATAL: Could not configure Gemini. {e}")

tavily_client = None
try:
    # Tavily API Key
    tavily_api_key = os.environ.get('TAVILY_API_KEY')
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in environment secrets.")
    tavily_client = TavilyClient(api_key=tavily_api_key)
    print("Tavily API client configured successfully.")
except Exception as e:
    print(f"FATAL: Could not configure Tavily. {e}")
    tavily_client = None


# --- AI Helper Functions ---

def search_real_time_info(query):
    """Perform real-time search for any query using Tavily."""
    if not tavily_client:
        # Caller should handle this gracefully; do not pass this string into the model.
        return {"error": "Real-time search unavailable - Tavily client not configured."}

    try:
        print(f"DEBUG: Performing real-time search for: {query}")
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=10,
            include_raw_content=True
            # Removed include_domains to allow general queries (sports/news/links/etc.)
        )

        if response and 'results' in response:
            results_with_urls = []
            for res in response['results']:
                content = res.get('raw_content') or res.get('content', '')
                url = res.get('url', '')
                title = res.get('title', '')
                if url:
                    results_with_urls.append({
                        'title': title or url,
                        'url': url,
                        'content': (content[:500] + '...') if content and len(content) > 500 else (content or '')
                    })

            if results_with_urls:
                return {"results": results_with_urls[:5]}
            else:
                return {"results": []}
        else:
            return {"results": []}

    except Exception as e:
        print(f"ERROR: Real-time search failed: {e}")
        return {"error": f"Real-time search error: {e}"}


def generate_initial_ideas(domain, challenge, skills, url):
    """Generates initial hackathon ideas, enhanced by scraping a provided URL."""
    hackathon_context = "Not provided or could not be fetched."
    if url and tavily_client:
        print(f"DEBUG: Attempting to fetch content from URL: {url}")
        try:
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
        if not google_configured:
            raise RuntimeError("Gemini is not configured. Missing or invalid GOOGLE_API_KEY.")

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
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json", "response_schema": json_schema}
        )
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
        ideas_json['hackathon_context'] = hackathon_context
        ideas_json['tavily_used'] = bool(url and tavily_client and "could not be fetched" not in hackathon_context.lower())
        ideas_json['has_real_time_access'] = bool(tavily_client)
        return ideas_json
    except Exception as e:
        print(f"ERROR: Error in generate_initial_ideas: {e}")
        return {"error": "Could not generate initial ideas.", "details": str(e)}


def should_perform_search(query: str) -> bool:
    """Deterministically decide if a web search is required based on the query text."""
    if not query:
        return False

    q = query.lower()

    keywords = [
        "link", "links", "url", "site", "website", "where can i",
        "find me", "current", "latest", "recent", "today", "now",
        "news", "who won", "score", "schedule", "release", "announced",
        "launched", "deadline", "date", "price", "pricing", "docs",
        "documentation", "tutorial", "repo", "github", "hackathon",
        "event", "conference"
    ]
    if any(k in q for k in keywords):
        return True

    if re.search(r"\b20(2[0-9]|3[0-9])\b", q):
        return True

    if q.strip().endswith("?") and any(w in q for w in ["who", "when", "where", "what’s new", "what is new"]):
        return True

    return False


def generate_chat_response(history):
    """Generates a contextual chat response based on the conversation history."""
    try:
        if not google_configured:
            raise RuntimeError("Gemini is not configured. Missing or invalid GOOGLE_API_KEY.")

        initial_ideas_obj = json.loads(history[1]['content']) if isinstance(history[1]['content'], str) else history[1]['content']
        scraped_context = initial_ideas_obj.get('hackathon_context', 'No web context was available.')
        
        chat_turns = []
        for msg in history[2:]:
            role = "model" if msg['role'] == 'assistant' else 'user'
            chat_turns.append({"role": role, "parts": [msg['content']]})

        if not chat_turns:
            return {"error": "No follow-up question found in history."}

        latest_question = chat_turns.pop()['parts'][0]
        
        needs_real_time = should_perform_search(latest_question)
        
        search_results_for_ai = ""
        used_real_time_search = False

        if needs_real_time:
            print(f"DEBUG: Real-time search triggered for question: {latest_question}")
            search_payload = search_real_time_info(latest_question)

            if isinstance(search_payload, dict) and "results" in search_payload and search_payload["results"]:
                used_real_time_search = True
                formatted_results = []
                for i, item in enumerate(search_payload["results"], 1):
                    title = item.get("title") or "Untitled"
                    url = item.get("url") or ""
                    snippet = item.get("content") or ""
                    formatted_results.append(f"{i}. **{title}**\nURL: {url}\nContent: {snippet}\n")
                search_results_for_ai = "\n\n**LIVE SEARCH RESULTS:**\n" + "\n".join(formatted_results) + "\n**END OF SEARCH RESULTS**"
            else:
                print("DEBUG: No usable real-time results; proceeding without injecting results.")
                used_real_time_search = False

        system_instruction = f"""
You are "SYNAPSE AI," a highly intelligent AI assistant from SYNAPSE AI LTD with LIVE REAL-TIME WEB SEARCH capabilities.

CRITICAL RULES:
1. When search results are provided, use them and surface specific links.
2. Format all URLs as clickable markdown links: [Link Text](https://example.com).
3. Be concise and helpful.

CONTEXT FOR THIS CONVERSATION:
- Scraped context from earlier: {scraped_context}

CURRENT USER REQUEST ANALYSIS:
- Real-time search was {"PERFORMED" if used_real_time_search else "NOT PERFORMED"} for this query.

{search_results_for_ai}

INSTRUCTION: Use the search results above if present to answer the user's question with specific links and current information. If no results are present, answer with your best knowledge and suggest precise next steps to refine the search.
"""

        model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_instruction)
        chat = model.start_chat(history=chat_turns)
        response = chat.send_message(latest_question)
        
        return {
            "response": response.text,
            "used_real_time_search": used_real_time_search,
            "search_triggered": needs_real_time
        }
    except Exception as e:
        print(f"Error in generate_chat_response: {e}")
        return {"error": "Could not generate chat response.", "details": str(e)}


def generate_pitch(history):
    try:
        if not google_configured:
            raise RuntimeError("Gemini is not configured. Missing or invalid GOOGLE_API_KEY.")

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
    try:
        if not google_configured:
            raise RuntimeError("Gemini is not configured. Missing or invalid GOOGLE_API_KEY.")

        json_schema = {
            "type": "object",
            "properties": {
                "teammates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "reason": {"type": "string"}
                        },
                        "required": ["role", "reason"]
                    }
                }
            }
        }
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json", "response_schema": json_schema}
        )
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
    ideas = generate_initial_ideas(
        data.get('domain'),
        data.get('challenge'),
        data.get('skills'),
        data.get('hackathon_url')
    )
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
    id_token = request.headers.get('Authorization', '').split('Bearer ')[-1]
    if not id_token:
        raise ValueError("Authorization token not found.")
    decoded_token = auth.verify_id_token(id_token)
    return decoded_token['uid']

@app.route('/save_session', methods=['POST'])
def save_session():
    if not db:
        return jsonify({"error": "Firestore is not configured."}), 500
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
    if not db:
        return jsonify({"error": "Firestore is not configured."}), 500
    try:
        uid = _get_user_id_from_token(request)
        sessions_ref = db.collection('users').document(uid).collection('sessions') \
            .order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        sessions = []
        for session in sessions_ref:
            s_data = session.to_dict()
            title = "New Brainstorm"
            if s_data.get('history') and len(s_data['history']) > 0:
                title = s_data['history'][0].get('content', "New Brainstorm")
            sessions.append({"id": session.id, "title": title})
        return jsonify(sessions)
    except Exception as e:
        return jsonify({"error": str(e)}), 401

@app.route('/get_session/<session_id>', methods=['GET'])
def get_session(session_id):
    if not db:
        return jsonify({"error": "Firestore is not configured."}), 500
    try:
        uid = _get_user_id_from_token(request)
        doc = db.collection('users').document(uid).collection('sessions').document(session_id).get()
        return jsonify(doc.to_dict()) if doc.exists else (jsonify({"error": "Session not found."}), 404)
    except Exception as e:
        return jsonify({"error": str(e)}), 401

@app.route('/delete_session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    if not db:
        return jsonify({"error": "Firestore is not configured."}), 500
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

# --- Health Check ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "env": {
            "GOOGLE_API_KEY": bool(os.environ.get('GOOGLE_API_KEY')),
            "TAVILY_API_KEY": bool(os.environ.get('TAVILY_API_KEY')),
            "FIREBASE_SERVICE_ACCOUNT_KEY": bool(os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY'))
        },
        "configured": {
            "gemini": google_configured,
            "tavily": tavily_client is not None,
            "firestore": db is not None
        }
    })

# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Set debug as needed; keep False in production
    app.run(host='0.0.0.0', port=port, debug=False)
