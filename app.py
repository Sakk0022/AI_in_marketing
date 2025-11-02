from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import google.generativeai as genai
import requests
import re
from sentence_transformers import SentenceTransformer
import chromadb
from deep_translator import GoogleTranslator
from lingua import Language, LanguageDetectorBuilder
import json
from bs4 import BeautifulSoup
from pulp import *
import tempfile
# === Настройки ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)
app.secret_key = os.urandom(24)
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in environment variables!")

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
# Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
# Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set!")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
# === ChromaDB: в памяти (Render не сохраняет файлы) ===
temp_dir = tempfile.mkdtemp()
chroma_client = chromadb.PersistentClient(path=temp_dir)
collection = chroma_client.get_or_create_collection(name="ad_docs")
# Документы
documents = [
    "Ad text must not promote illegal activities, scams, hate speech, or violence.",
    "Ensure ads are truthful, not misleading, and comply with platform policies.",
    "Creative ad example for phones: 'Unlock the future with our lightning-fast smartphones – speed that thrills!'",
    "Varied ad for tech: 'Experience innovation in your pocket: Sleek design meets powerful performance.'",
    "Compliance: Avoid claims without evidence, like 'best in the world' unless proven.",
    "Interesting ad variation: 'Tired of boring calls? Our phones turn conversations into adventures!'"
]
# Инициализация при старте
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(documents).tolist()
    ids = [f"doc{i}" for i in range(len(documents))]
    if collection.count() == 0:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings
        )
    print("ChromaDB initialized.")
except Exception as e:
    print(f"ChromaDB init failed: {e}")
# === Модели ===
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
class Campaign(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    budget = db.Column(db.Float, nullable=False, default=0.0)
    ad_text = db.Column(db.Text)
    platforms = db.Column(db.String(200))
    status = db.Column(db.String(50), default='draft')
    budget_distribution = db.Column(db.Text)
    real_time_data = db.Column(db.Text)
    ab_testing_plans = db.Column(db.Text)
with app.app_context():
    db.create_all()
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
# === Язык ===
language_detector = LanguageDetectorBuilder.from_all_languages().build()
def detect_language(message):
    try:
        if not message.strip(): return 'en'
        detected = language_detector.detect_language_of(message)
        return detected.iso_code_639_1.name.lower() if detected else 'en'
    except:
        return 'en'
def translate_text(text, target_lang):
    if target_lang == 'en': return text
    translator = GoogleTranslator(source='en', target=target_lang)
    max_length = 4000
    if len(text) <= max_length:
        return translator.translate(text)
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
    return ''.join(translator.translate(chunk) for chunk in chunks)
def translate_to_english(message, source_lang):
    if source_lang == 'en': return message
    return GoogleTranslator(source=source_lang, target='en').translate(message)
# === Вспомогательные функции ===
def jaccard_similarity(t1, t2):
    s1, s2 = set(t1.lower().split()), set(t2.lower().split())
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0
def check_plagiarism(text, known=[]):
    return any(jaccard_similarity(text, k) > 0.8 for k in known) and "Regenerating..." or text
def rag_query(query, top_k=3):
    q_emb = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    return "\n".join(results['documents'][0])
def generate_ad_text(prompt, known=[]):
    context = rag_query(prompt + " creative ad examples")
    text = model.generate_content(f"Generate creative ad: {prompt}. Context: {context}").text
    return check_plagiarism(text, known) if "Regenerating" not in check_plagiarism(text, known) else model.generate_content(f"Regenerate: {prompt}").text
def distribute_budget(total, platforms, msg):
    rois = {p: 2.5 for p in platforms}
    msg_l = msg.lower()
    for p in platforms:
        if p.lower() in ['instagram', 'tiktok'] and ('visual' in msg_l or 'social' in msg_l): rois[p] += 0.5
        elif p.lower() == 'google' and ('search' in msg_l or 'seo' in msg_l): rois[p] += 0.5
        elif p.lower() == 'email' and ('email' in msg_l): rois[p] += 0.5
    prob = LpProblem("Budget", LpMaximize)
    alloc = LpVariable.dicts("Alloc", platforms, lowBound=0)
    prob += lpSum(alloc[p] * rois[p] for p in platforms)
    prob += lpSum(alloc[p] for p in platforms) == total
    min_b, max_b = total * 0.1, total * 0.6
    for p in platforms:
        prob += alloc[p] >= (min_b if len(platforms) > 1 else 0)
        prob += alloc[p] <= max_b
    prob.solve()
    dist = {p: alloc[p].varValue for p in platforms}
    reasons = model.generate_content(f"Explain budget: {json.dumps(dist)}").text[:500]
    return dist, reasons, ""
def compliance_scan(text):
    issues = model.generate_content(f"Scan ad: {text}. Rules: {rag_query('compliance')}").text
    return issues if "issue" in issues.lower() else None
def fetch_real_time_data(query):
    topic = re.search(r'(?:for|about|on)\s+(.+)', query, re.I)
    q = (topic.group(1) if topic else query).strip().rstrip('s')
    try:
        r = requests.get(f"https://en.wikipedia.org/wiki/{q.replace(' ', '_')}", timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        if r.ok:
            soup = BeautifulSoup(r.text, 'html.parser')
            return f"### Wikipedia\n" + ' '.join(p.text for p in soup.find_all('p')[:3])[:500]
    except: pass
    return "No data found."
# === Роуты ===
@app.route('/')
def home():
    return render_template('main.html', page='home')
@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json.get('message')
    lang = detect_language(msg)
    en_msg = translate_to_english(msg, lang)
    resp = ""
    key = None
    budget = None
    plats = None
    lower = en_msg.lower()
    if 'create campaign' in lower:
        name = re.search(r'create campaign\s*(?:named)?\s*(.+)', en_msg, re.I)
        name = (name.group(1) if name else "New").strip()
        if current_user.is_authenticated:
            c = Campaign(user_id=current_user.id, name=name)
            db.session.add(c); db.session.commit()
            session['current_campaign_id'] = c.id
            resp = f"Campaign '{name}' created."
        else:
            resp = "Login required."
    elif 'switch campaign' in lower:
        m = re.search(r'switch campaign\s*(?:to)?\s*(\d+|\w+)', en_msg, re.I)
        if m and current_user.is_authenticated:
            iden = m.group(1)
            c = Campaign.query.filter_by(id=int(iden) if iden.isdigit() else None, user_id=current_user.id).first() or 
                Campaign.query.filter_by(name=iden, user_id=current_user.id).first()
            if c:
                session['current_campaign_id'] = c.id
                resp = f"Switched to '{c.name}'."
            else:
                resp = "Not found."
        else:
            resp = "Specify ID/name."
    else:
        if any(x in lower for x in ['ad text', 'generate ad']):
            text = generate_ad_text(en_msg)
            issues = compliance_scan(text)
            resp = f"### Ad Text\n{text}\n\n" + (f"### Issues\n{issues}" if issues else "**Compliant!**")
            key = 'ad_text'
        elif any(x in lower for x in ['budget', 'distribute']):
            b_match = re.search(r'\b\d[\d.]*\b', en_msg)
            budget = float(b_match.group()) if b_match else None
            if not budget:
                resp = "No budget."
            else:
                after = en_msg.split(str(budget))[-1]
                plats_part = after.split('on')[-1] if 'on' in after else after
                plats = [p.capitalize() for p in re.split(r'[,\s]+', plats_part) if p.lower() in ['google','meta','tiktok','instagram','facebook','youtube','email','fb','yt','ig','insta','tt']]
                if not plats: plats = ['Google','Meta','TikTok']
                dist, reasons, _ = distribute_budget(budget, plats, en_msg)
                resp = f"### Budget\n" + "\n".join(f"- **{p}**: ${v:.2f}" for p,v in dist.items()) + f"\n\n### Reasons\n{reasons}"
                key = 'budget_distribution'
        elif any(x in lower for x in ['data', 'news']):
            resp = fetch_real_time_data(en_msg)
            key = 'real_time_data'
        elif any(x in lower for x in ['a/b', 'scenarios']):
            resp = model.generate_content(f"3 A/B tests for: {en_msg}").text
            key = 'ab_testing_plans'
        else:
            resp = model.generate_content(f"{en_msg}\nContext: {rag_query(en_msg)}").text
        if key and current_user.is_authenticated and (cid := session.get('current_campaign_id')):
            c = Campaign.query.get(cid)
            if c and c.user_id == current_user.id:
                if key == 'budget_distribution' and budget and plats:
                    c.budget = budget
                    c.platforms = ','.join(plats)
                setattr(c, key, resp)
                db.session.commit()
    return jsonify({'response': translate_text(resp, lang)})
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('main.html', page='dashboard', campaigns=Campaign.query.filter_by(user_id=current_user.id).all())
@app.route('/download_row/<int:cid>')
@login_required
def download_row(cid):
    c = Campaign.query.get_or_404(cid)
    if c.user_id != current_user.id: return "Unauthorized", 403
    content = f"""=== Campaign ===
ID: {c.id} | Name: {c.name} | Budget: ${c.budget:.2f}
Ad: {c.ad_text or 'N/A'}
Platforms: {c.platforms or 'N/A'}
Distribution: {c.budget_distribution or 'N/A'}
Data: {c.real_time_data or 'N/A'}
A/B: {c.ab_testing_plans or 'N/A'}
"""
    return Response(content, mimetype='text/plain', headers={"Content-Disposition": f"attachment; filename=campaign_{cid}.txt"})
@app.route('/compliance_check', methods=['GET', 'POST'])
def compliance_check():
    result = None
    if request.method == 'POST':
        text = request.form.get('ad_text')
        lang = detect_language(text)
        issues = compliance_scan(translate_to_english(text, lang))
        result = translate_text(f"### Issues\n{issues}" if issues else "**Compliant!**", lang)
    return render_template('main.html', page='compliance', result=result)
# === Авторизация ===
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        u = User(username=request.form['username'], password=request.form['password'], email=request.form['email'])
        db.session.add(u); db.session.commit()
        return redirect(url_for('login'))
    return render_template('main.html', page='register')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.password == request.form['password']:
            login_user(user)
            return redirect(url_for('home'))
    return render_template('main.html', page='login')
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


