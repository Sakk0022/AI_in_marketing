from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import google.generativeai as genai
import requests
import re
import json
from bs4 import BeautifulSoup
from pulp import *
from deep_translator import GoogleTranslator
from lingua import Language, LanguageDetectorBuilder

# === Настройки ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Убираем предупреждение

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

if not app.config['SQLALCHEMY_DATABASE_URI']:
    raise ValueError("DATABASE_URL is not set!")

db = SQLAlchemy(app)

# === Flask-Login ===
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# === Gemini ===
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set!")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')  # Updated to support structured outputs better

# Setup for RAG: Vector Database with ChromaDB and Sentence Transformers (embeddings via transformer model)
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # File-based vector DB
collection = chroma_client.get_or_create_collection(name="ad_docs")

# Pre-load some example documents for RAG (compliance rules, ad examples for variety)
# In production, load from files or DB
documents = [
    "Ad text must not promote illegal activities, scams, hate speech, or violence.",
    "Ensure ads are truthful, not misleading, and comply with platform policies.",
    "Creative ad example for phones: 'Unlock the future with our lightning-fast smartphones – speed that thrills!'",
    "Varied ad for tech: 'Experience innovation in your pocket: Sleek design meets powerful performance.'",
    "Compliance: Avoid claims without evidence, like 'best in the world' unless proven.",
    "Interesting ad variation: 'Tired of boring calls? Our phones turn conversations into adventures!'"
]
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Transformer-based embedding model
embeddings = embedder.encode(documents)
ids = [f"doc{i}" for i in range(len(documents))]
collection.add(documents=documents, embeddings=embeddings.tolist(), ids=ids)

# Database Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)  # In production, hash passwords
    email = db.Column(db.String(120), unique=True, nullable=False)

class Campaign(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    budget = db.Column(db.Float, nullable=False, default=0.0)
    ad_text = db.Column(db.Text)
    platforms = db.Column(db.String(200))  # e.g., 'Google,Meta,TikTok'
    status = db.Column(db.String(50), default='draft')
    budget_distribution = db.Column(db.Text)  # JSON string for distribution
    real_time_data = db.Column(db.Text)
    ab_testing_plans = db.Column(db.Text)

# Create database tables if they don't exist (no drop to preserve data)
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Helper Functions

# Создаем детектор языка один раз для эффективности
language_detector = LanguageDetectorBuilder.from_all_languages().build()

def detect_language(message):
    """Detect language of the message using lingua for high accuracy"""
    try:
        if not message.strip():
            return 'en'
        detected = language_detector.detect_language_of(message)
        if detected:
            return detected.iso_code_639_1.name.lower()
        else:
            return 'en'  # Fallback to English if not detected
    except:
        return 'en'

def translate_text(text, target_lang):
    """Translate text to target language, handling long text by splitting"""
    if target_lang == 'en':
        return text
    translator = GoogleTranslator(source='en', target=target_lang)
    max_length = 4000  # Safe limit below 5000
    if len(text) <= max_length:
        return translator.translate(text)
    else:
        # Split into chunks
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        translated_chunks = [translator.translate(chunk) for chunk in chunks]
        return ''.join(translated_chunks)

def translate_to_english(message, source_lang):
    """Translate to English if not English"""
    if source_lang != 'en':
        translator = GoogleTranslator(source=source_lang, target='en')
        return translator.translate(message)
    return message

def jaccard_similarity(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0

def check_plagiarism(text, known_texts=[]):
    for known in known_texts:
        if jaccard_similarity(text, known) > 0.8:
            return "Possible plagiarism detected. Regenerating..."
    return text

def rag_query(query, top_k=3):
    """Retrieval Augmented Generation: Retrieve relevant docs from vector DB"""
    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    retrieved_docs = results['documents'][0]
    return "\n".join(retrieved_docs)

def generate_ad_text(prompt, known_texts=[]):
    """AI-driven ad text creation using Gemini with RAG for variety and compliance"""
    rag_context = rag_query(prompt + " creative ad examples")
    full_prompt = f"Generate creative, interesting, and varied ad text based on: {prompt}. Use this context for inspiration and compliance: {rag_context}"
    response = model.generate_content(full_prompt)
    text = response.text
    plag_check = check_plagiarism(text, known_texts)
    if "plagiarism" in plag_check:
        # Regenerating
        response = model.generate_content(full_prompt)
        text = response.text
    return text

def distribute_budget(total_budget, platforms, user_message):
    """Intelligent budget distribution using PuLP optimization"""
    # Adjust ROIs based on user message (similar to adjust_rois)
    rois = {platform: 2.5 for platform in platforms}  # Default ROI
    message_lower = user_message.lower()
    for platform in platforms:
        if platform.lower() in ['instagram', 'tiktok'] and ('visual' in message_lower or 'social media' in message_lower):
            rois[platform] += 0.5
        elif platform.lower() == 'google' and ('search' in message_lower or 'seo' in message_lower):
            rois[platform] += 0.5
        elif platform.lower() == 'email' and ('email' in message_lower or 'newsletter' in message_lower):
            rois[platform] += 0.5

    # PuLP optimization
    prob = LpProblem("Budget_Allocation", LpMaximize)
    allocations = LpVariable.dicts("Alloc", platforms, lowBound=0)
    
    prob += lpSum([allocations[p] * rois[p] for p in platforms])
    
    prob += lpSum([allocations[p] for p in platforms]) == total_budget
    
    min_budget = total_budget * 0.1
    max_budget = total_budget * 0.6
    for p in platforms:
        prob += allocations[p] >= min_budget if len(platforms) > 1 else 0, f"Min_budget_{p}"
        prob += allocations[p] <= max_budget, f"Max_budget_{p}"
    
    prob.solve()
    distribution = {p: allocations[p].varValue for p in platforms}
    
    # Use AI for reasons and suggestions, keeping it concise
    analysis_prompt = f"Provide short reasons (1-2 sentences) for this budget distribution: {json.dumps(distribution)} for request: '{user_message}'. Also, short suggestions (1-2 sentences per platform) on what to do with the money on each platform."
    response = model.generate_content(analysis_prompt)
    try:
        output = json.loads(response.text.strip())
        reasons = output.get('reasons', 'No reasons provided.')
        suggestions = output.get('suggestions', 'No suggestions provided.')
    except:
        reasons = response.text[:1000]  # Truncate if needed
        suggestions = ''

    return distribution, reasons, suggestions

def compliance_scan(text):
    """Legal/ethical compliance scan with RAG (retrieve rules and check)"""
    rag_context = rag_query("compliance rules for ads")
    full_prompt = f"Scan this ad text for legal/ethical issues: {text}. Use these rules: {rag_context}. List issues if any."
    response = model.generate_content(full_prompt)
    issues = response.text
    return issues if "issues" in issues.lower() else None

def fetch_real_time_data(query):
    """Real-time internet data integration using Wikipedia/DuckDuckGo"""
    # Improved topic extraction
    topic_match = re.search(r'(?:for|about|on|data\s+for)\s+(.+)', query, re.IGNORECASE)
    search_query = topic_match.group(1).strip() if topic_match else query.strip()
    # Normalize to singular for Wikipedia
    if search_query.endswith('s'):
        search_query = search_query[:-1]
    
    print(f"Fetching web info for: {search_query}")
    url = f"https://en.wikipedia.org/wiki/{search_query.replace(' ', '_')}"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'}
    try:
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        info = ' '.join([p.text for p in paragraphs[:3]])
        print("Web info fetched from Wikipedia")
        formatted = f"### Real-Time Data from Wikipedia\n{info.strip()[:500]}"
        return formatted
    except Exception as e:
        print(f"Wikipedia error: {e}")
        search_url = f"https://duckduckgo.com/html/?q={search_query}"
        try:
            response = requests.get(search_url, timeout=10, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            results = soup.find_all('div', class_='result__body')
            info = ' '.join([r.text for r in results[:2]])
            print("Web info fetched from DuckDuckGo")
            formatted = f"### Real-Time Data from DuckDuckGo\n{info.strip()[:500]}"
            return formatted
        except Exception as e2:
            print(f"DuckDuckGo error: {e2}")
            return "Relevant information about the query."

def plan_ab_testing(prompt, known_texts=[]):
    """Scenario planning for A/B testing: Generate detailed scenarios using Gemini"""
    full_prompt = f"Generate 3 detailed A/B testing scenarios for ad campaign based on: {prompt}. For each scenario, include: variations in text, targeting, budget split, expected metrics, and rationale. Format as Markdown with headings for each scenario."
    response = model.generate_content(full_prompt)
    text = response.text
    plag_check = check_plagiarism(text, known_texts)
    if "plagiarism" in plag_check:
        # Regenerating
        response = model.generate_content(full_prompt)
        text = response.text
    return text

# Routes

@app.route('/')
def home():
    """Home page with chat interface"""
    return render_template('main.html', page='home')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and process features"""
    user_message = request.json.get('message')
    user_lang = detect_language(user_message)
    translated_message = translate_to_english(user_message, user_lang)
    response_en = ""
    save_key = None
    total_budget = None
    platforms = None

    # Detect intent and route to features (improved detection)
    lower_msg = translated_message.lower()
    if 'create campaign' in lower_msg:
        name_match = re.search(r'create campaign\s*(named)?\s*(.+)', translated_message, re.IGNORECASE)
        name = name_match.group(2).strip() if name_match else "New Campaign"
        if current_user.is_authenticated:
            new_campaign = Campaign(
                user_id=current_user.id,
                name=name,
                budget=0.0,
                ad_text="",
                platforms="",
                budget_distribution="",
                real_time_data="",
                ab_testing_plans=""
            )
            db.session.add(new_campaign)
            db.session.commit()
            session['current_campaign_id'] = new_campaign.id
            response_en = f"Campaign '{name}' created."
        else:
            response_en = "Please login to create campaigns."
    elif 'switch campaign' in lower_msg:
        # Parse id or name
        match = re.search(r'switch campaign\s*(to)?\s*(\d+|\w+.+)', translated_message, re.IGNORECASE)
        if match:
            identifier = match.group(2).strip()
            if current_user.is_authenticated:
                if identifier.isdigit():
                    campaign = Campaign.query.filter_by(id=int(identifier), user_id=current_user.id).first()
                else:
                    campaign = Campaign.query.filter_by(name=identifier, user_id=current_user.id).first()
                if campaign:
                    session['current_campaign_id'] = campaign.id
                    response_en = f"Switched to campaign '{campaign.name}'."
                else:
                    response_en = "Campaign not found."
            else:
                response_en = "Please login."
        else:
            response_en = "Please specify campaign name or ID."
    else:
        if 'ad text' in lower_msg or 'generate ad' in lower_msg:
            ad_text = generate_ad_text(translated_message)
            issues = compliance_scan(ad_text)
            if issues:
                response_en = f"### Generated Ad Text\n{ad_text}\n\n### Compliance Issues\n{issues}"
            else:
                response_en = f"### Generated Ad Text\n{ad_text}\n\n**Compliant!**"
            save_key = 'ad_text'
        elif 'budget' in lower_msg or 'distribute' in lower_msg:
            try:
                # Improved parsing
                budget_match = re.search(r'\b\d+[\.\d]*\b', translated_message)
                total_budget = float(budget_match.group(0)) if budget_match else None
                if not total_budget:
                    raise ValueError("No budget found")
                
                after_budget = translated_message.split(str(total_budget))[-1].strip()
                if 'on' in after_budget:
                    platforms_part = after_budget.split('on')[-1].strip()
                else:
                    platforms_part = after_budget
                
                potential = re.split(r'[,\s]+', platforms_part)
                platforms = [p.capitalize() for p in potential if p and p.lower() in ['google', 'meta', 'tiktok', 'instagram', 'facebook', 'youtube', 'email', 'fb', 'yt', 'ig', 'insta', 'tt']]
                if not platforms:
                    platforms = ['Google', 'Meta', 'TikTok']  # Default
                
                distribution, reasons, suggestions = distribute_budget(total_budget, platforms, translated_message)
                formatted_dist = "\n".join([f"- **{platform}**: ${amount:.2f}" for platform, amount in distribution.items()])
                response_en = f"### Budget Distribution\n{formatted_dist}\n\n### Reasons\n{reasons}\n\n### Suggestions\n{suggestions}"
                save_key = 'budget_distribution'
            except ValueError as ve:
                response_en = str(ve)
            except:
                response_en = "Please provide a valid budget and platforms (e.g., Distribute 10000 on Google, Meta, TikTok)"
        elif 'data' in lower_msg or 'news' in lower_msg:
            data = fetch_real_time_data(translated_message)
            response_en = data
            save_key = 'real_time_data'
        elif 'a/b testing' in lower_msg or 'scenarios' in lower_msg:
            plans = plan_ab_testing(translated_message)
            response_en = f"### A/B Testing Plans\n{plans}"
            save_key = 'ab_testing_plans'
        else:
            # Fallback to general Gemini response with RAG if relevant
            rag_context = rag_query(translated_message)
            gemini_response = model.generate_content(f"{translated_message}\nContext: {rag_context}")
            response_en = gemini_response.text

        # Save to campaign if applicable and user is authenticated
        if save_key and current_user.is_authenticated:
            current_id = session.get('current_campaign_id')
            if current_id:
                campaign = Campaign.query.get(current_id)
                if campaign and campaign.user_id == current_user.id:
                    if save_key == 'budget_distribution' and total_budget is not None and platforms is not None:
                        campaign.budget = total_budget
                        campaign.platforms = ','.join(platforms)
                    setattr(campaign, save_key, response_en)  # Save the English response? Or translated? Wait, response_en is English
                    db.session.commit()
                else:
                    response_en += "\nInvalid current campaign."
            else:
                response_en += "\nPlease create or switch to a campaign first."

    # Translate response back to user language
    response = translate_text(response_en, user_lang)

    return jsonify({'response': response})

@app.route('/dashboard')
@login_required
def dashboard():
    """Live process dashboard - now with real data from DB"""
    campaigns = Campaign.query.filter_by(user_id=current_user.id).all()
    return render_template('main.html', page='dashboard', campaigns=campaigns)

@app.route('/download_row/<int:campaign_id>')
@login_required
def download_row(campaign_id):
    campaign = Campaign.query.get_or_404(campaign_id)
    if campaign.user_id != current_user.id:
        return "Unauthorized", 403

    # Формируем содержимое строки в читаемом виде
    content = f"""=== Campaign Row Export ===
ID: {campaign.id}
Name: {campaign.name}
Budget: ${campaign.budget:.2f}
Ad Text: {campaign.ad_text or 'N/A'}
Platforms: {campaign.platforms or 'N/A'}
Status: {campaign.status}
Budget Distribution:
{campaign.budget_distribution or 'N/A'}

Real-Time Data:
{campaign.real_time_data or 'N/A'}

A/B Testing Plans:
{campaign.ab_testing_plans or 'N/A'}
"""

    filename = f"campaign_row_{campaign.id}_{campaign.name.replace(' ', '_')}.txt"
    return Response(
        content,
        mimetype='text/plain',
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.route('/compliance_check', methods=['GET', 'POST'])
def compliance_check():
    """New page for user to input ad text and check compliance"""
    result = None
    if request.method == 'POST':
        ad_text = request.form.get('ad_text')
        user_lang = detect_language(ad_text)
        translated_text = translate_to_english(ad_text, user_lang)
        issues = compliance_scan(translated_text)
        formatted_issues_en = f"### Compliance Issues\n{issues}" if issues else "**Compliant!**"
        result = translate_text(formatted_issues_en, user_lang)
    return render_template('main.html', page='compliance', result=result)

# Authentication Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']  # Hash in production
        email = request.form['email']
        new_user = User(username=username, password=password, email=email)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('main.html', page='register')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:  # Check hash in production
            login_user(user)
            return redirect(url_for('home'))
    return render_template('main.html', page='login')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))
    

