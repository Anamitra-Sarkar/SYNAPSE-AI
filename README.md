SYNAPSE AI - Hackathon Brainstorming Agent
SYNAPSE AI is an intelligent, AI-powered brainstorming partner designed to help developers and innovators excel in hackathons. By leveraging the power of Google's Gemini-2.5-Pro and the Tavily real-time search API, this application provides insightful project ideas, finds current hackathons, and even helps build the perfect team.

✨ Features
🧠 AI-Powered Brainstorming: Input your domain, a hackathon's theme, and your personal skills to receive a curated list of problem statements and detailed project concepts.

🌐 Real-Time Web Search: Ask for current or upcoming hackathons, and SYNAPSE AI will use its real-time search capabilities to find relevant links and information.

📄 Dynamic Conversation: Engage in a natural conversation to refine ideas, ask follow-up questions, and explore different concepts.

🚀 Elevator Pitch Generation: With a single click, generate a compelling 30-second elevator pitch for your project idea.

🤝 AI Team Finder: Get intelligent suggestions for ideal teammates with complementary skills to round out your hackathon team.

🔐 Secure & Persistent: Features full user authentication (Email/Password & Google) and saves your brainstorming sessions to the cloud with Firebase Firestore.

🎨 Sleek & Responsive UI: A modern, dark-themed interface built with Tailwind CSS that works beautifully on both desktop and mobile devices.

🛠️ Tech Stack
Backend: Python with Flask

AI & Search:

Google Gemini-2.5-Pro for core generative AI capabilities.

Tavily Search API for real-time web crawling and information retrieval.

Frontend: HTML, CSS, and modern JavaScript (ESM)

Styling: Tailwind CSS for a utility-first design approach.

Database & Auth: Google Firebase (Firestore for database, Authentication for user management).

Deployment: Configured for easy deployment on Render.

⚙️ Setup & Installation
To run this project locally, you will need to have Python and a virtual environment manager installed.

1. Clone the Repository
git clone https://github.com/Anamitra-Sarkar/SYNAPSE-AI.git
cd synapse-ai

2. Set Up the Backend
Create and activate a Python virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required Python packages:

pip install -r requirements.txt

3. Configure Environment Variables
You will need to set up API keys for the services used in this project. Create a .env file in the root of your project directory and add the following:

# Your Google API key for Gemini
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

# Your Tavily API key for real-time search
TAVILY_API_KEY="YOUR_TAVILY_API_KEY"

# Your Firebase Service Account Key (as a single-line JSON string)
FIREBASE_SERVICE_ACCOUNT_KEY='{"type": "service_account", "project_id": "...", ...}'

Important: To get your FIREBASE_SERVICE_ACCOUNT_KEY, go to your Firebase project settings, then "Service accounts," and generate a new private key. Open the downloaded JSON file and copy its contents as a single line.

4. Configure the Frontend
Open the index.html file.

Find the firebaseConfig object in the <script type="module"> section.

Replace the placeholder values with your own Firebase project's web app configuration.

5. Run the Application
Start the Flask server from the root directory:

flask run

Open your browser and navigate to http://127.0.0.1:5000.

🚀 Deployment
This application is configured for deployment on Render.

Push to GitHub: Ensure your project is pushed to a GitHub repository.

Create a New Web Service on Render:

Connect your GitHub account to Render.

Select your repository.

Build Command: pip install -r requirements.txt

Start Command: gunicorn app:app

Add Environment Variables: In the Render dashboard, go to the "Environment" section and add the same GOOGLE_API_KEY, TAVILY_API_KEY, and FIREBASE_SERVICE_ACCOUNT_KEY that you used in your local .env file.

Deploy: Click "Create Web Service." Render will automatically build and deploy your application.
