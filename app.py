from flask import Flask, render_template, request, jsonify, session, redirect, url_for, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import google.generativeai as genai
import requests
import re
import json
from pulp import *
from deep_translator import GoogleTranslator
from lingua import Language, LanguageDetectorBuilder

# === Настройки ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
model = genai.GenerativeModel('gemini-2.5-flash')

# === Язык ===
language_detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.RUSSIAN, Language.KAZAKH).build()

def detect_language(text):
    if not text.strip(): return 'en'
    lang = language_detector.detect_language_of(text)
    return lang.iso_code_639_1.name.lower() if lang else 'en'

def translate_text(text, target):
    if target == 'en' or not text: return text
    return GoogleTranslator(source='en', target=target).translate(text)

def translate_to_english(text, src):
    if src == 'en': return text
    return GoogleTranslator(source=src, target='en').translate(text)

# === RAG ===
RAG_DOCS = [
    "Ad text must not promote illegal activities, scams, hate speech, or violence.",
    "Ensure ads are truthful, not misleading, and comply with platform policies.",
    "Creative ad example: 'Unlock the future with lightning-fast smartphones!'",
    "Varied ad: 'Experience innovation in your pocket.'",
    "Avoid claims like 'best in the world' without proof.",
    "Tired of boring calls? Our phones turn conversations into adventures!"
]

def rag_context(query):
    return "\n".join([doc for doc in RAG_DOCS if any(word in doc.lower() for word in query.lower().split())][:3]) or "Be creative and compliant."

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
    budget = db.Column(db.Float, default=0.0)
    ad_text = db.Column(db.Text)
    platforms = db.Column(db.String(200))
    budget_distribution = db.Column(db.Text)
    real_time_data = db.Column(db.Text)
    ab_testing_plans = db.Column(db.Text)
    status = db.Column(db.String(20), default='draft')  # Добавлено

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(uid):
    return User.query.get(int(uid))

# === Функции ===
def generate_ad_text(prompt):
    context = rag_context(prompt)
    resp = model.generate_content(f"Generate 1 creative, compliant ad: {prompt}\nContext: {context}")
    return resp.text.strip()

def distribute_budget(total, platforms, msg):
    rois = {p: 2.5 for p in platforms}
    msg_l = msg.lower()
    for p in platforms:
        if p.lower() in ['instagram', 'tiktok'] and 'visual' in msg_l: rois[p] += 0.7
        elif p.lower() == 'google' and 'search' in msg_l: rois[p] += 0.7
    prob = LpProblem("Budget", LpMaximize)
    x = LpVariable.dicts("x", platforms, lowBound=0)
    prob += lpSum(x[p] * rois[p] for p in platforms)
    prob += lpSum(x[p] for p in platforms) == total
    for p in platforms:
        prob += x[p] >= total * 0.1 if len(platforms) > 1 else 0
        prob += x[p] <= total * 0.6
    prob.solve(PULP_CBC_CMD(msg=0))
    dist = {p: round(x[p].value(), 2) for p in platforms}
    reasons = model.generate_content(f"Explain briefly: {json.dumps(dist)}").text[:300]
    return dist, reasons

def compliance_scan(text):
    resp = model.generate_content(f"Check ad for issues: {text}").text.lower()
    return resp if 'issue' in resp or 'violat' in resp else None

def fetch_real_time_data(query):
    q = re.sub(r'.*(?:for|on)\s+', '', query, flags=re.I).strip().rstrip('s')
    try:
        r = requests.get(f"https://en.wikipedia.org/w/api.php", params={
            'action': 'query', 'titles': q.replace(' ', '_'), 'prop': 'extracts',
            'exintro': True, 'explaintext': True, 'format': 'json'
        }, timeout=8)
        pages = r.json().get('query', {}).get('pages', {})
        text = next((v['extract'] for v in pages.values() if 'extract' in v), '')
        return f"### Wikipedia\n{text[:500]}" if text else "No data."
    except:
        return "No data."

# === Роуты ===
@app.route('/')
def home():
    return render_template('main.html', page='home')

@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json.get('message', '')
    lang = detect_language(msg)
    en_msg = translate_to_english(msg, lang)
    resp = ""
    key = None
    budget = None
    plats = None

    lower = en_msg.lower()

    # === CREATE CAMPAIGN ===
    if 'create campaign' in lower:
        name_match = re.search(r'create campaign\s*(?:named)?\s*(.+)', en_msg, re.I)
        name = name_match.group(1).strip() if name_match else "Untitled Campaign"
        name = name or "Untitled Campaign"
        if len(name) > 100:
            name = name[:97] + "..."

        if current_user.is_authenticated:
            c = Campaign(user_id=current_user.id, name=name, status='draft')
            db.session.add(c)
            db.session.commit()
            session['current_campaign_id'] = c.id
            resp = f"Campaign '{name}' created (ID: {c.id})."
        else:
            resp = "Login required."

    # === SWITCH CAMPAIGN ===
    elif 'switch campaign' in lower:
        match = re.search(r'switch\s+campaign\s+(.+)', en_msg, re.I)
        if match and current_user.is_authenticated:
            search_term = match.group(1).strip()
            c = None
            if search_term.isdigit():
                c = Campaign.query.filter_by(id=int(search_term), user_id=current_user.id).first()
            else:
                c = Campaign.query.filter(
                    Campaign.name.ilike(f"%{search_term}%"),
                    Campaign.user_id == current_user.id
                ).first()

            if c:
                session['current_campaign_id'] = c.id
                resp = f"Switched to '{c.name}' (ID: {c.id})."
            else:
                all_camps = Campaign.query.filter_by(user_id=current_user.id).all()
                if all_camps:
                    names = ", ".join([f"{camp.name} (ID: {camp.id})" for camp in all_camps])
                    resp = f"Not found. Available: {names}"
                else:
                    resp = "No campaigns yet. Create one first."
        else:
            resp = "Usage: `switch campaign <name or ID>`"

    # === OTHER COMMANDS ===
    else:
        if any(x in lower for x in ['ad text', 'generate ad']):
            text = generate_ad_text(en_msg)
            issues = compliance_scan(text)
            resp = f"### Ad Text\n{text}\n\n" + (f"### Issues\n{issues}" if issues else "**Compliant!**")
            key = 'ad_text'
        elif any(x in lower for x in ['budget', 'distribute']):
            b = re.search(r'\b\d[\d,.]*\b', en_msg)
            budget = float(b.group().replace(',', '')) if b else None
            if not budget:
                resp = "Enter budget (e.g., 10000)."
            else:
                plats_part = re.sub(r'.*?\bon\b', '', en_msg, flags=re.I)
                plats = [p.capitalize() for p in re.findall(r'\b(\w+)\b', plats_part) if p.lower() in ['google','meta','tiktok','instagram','facebook','youtube','email']]
                if not plats: plats = ['Google', 'Meta', 'TikTok']
                dist, reasons = distribute_budget(budget, plats, en_msg)
                resp = f"### Budget\n" + "\n".join(f"- **{p}**: ${v}" for p,v in dist.items()) + f"\n\n### Reasons\n{reasons}"
                key = 'budget_distribution'
        elif any(x in lower for x in ['data', 'news']):
            resp = fetch_real_time_data(en_msg)
            key = 'real_time_data'
        elif any(x in lower for x in ['a/b', 'test']):
            resp = model.generate_content(f"3 A/B tests for: {en_msg}").text
            key = 'ab_testing_plans'
        else:
            resp = model.generate_content(f"{en_msg}\nContext: {rag_context(en_msg)}").text

        # Сохранение в текущую кампанию
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
    campaigns = Campaign.query.filter_by(user_id=current_user.id).all()
    campaigns_data = [
        {
            'id': c.id,
            'name': c.name,
            'budget': c.budget if c.budget is not None else 0.0,
            'ad_text': c.ad_text or 'N/A',
            'platforms': c.platforms or 'N/A',
            'status': c.status or 'draft',
            'budget_distribution': c.budget_distribution or 'N/A',
            'real_time_data': c.real_time_data or 'N/A',
            'ab_testing_plans': c.ab_testing_plans or 'N/A',
        } for c in campaigns
    ]
    return render_template('main.html', page='dashboard', campaigns=campaigns_data)

@app.route('/download_row/<int:cid>')
@login_required
def download_row(cid):
    c = Campaign.query.get_or_404(cid)
    if c.user_id != current_user.id:
        return "Unauthorized", 403
    content = f"""=== Campaign ===
ID: {c.id} | Name: {c.name} | Budget: ${c.budget or 0.0}
Ad: {c.ad_text or 'N/A'}
Platforms: {c.platforms or 'N/A'}
Distribution: {c.budget_distribution or 'N/A'}
Data: {c.real_time_data or 'N/A'}
A/B: {c.ab_testing_plans or 'N/A'}
Status: {c.status or 'draft'}
"""
    return Response(content, mimetype='text/plain', headers={
        "Content-Disposition": f"attachment; filename=campaign_{cid}.txt"
    })

@app.route('/compliance_check', methods=['GET', 'POST'])
def compliance_check():
    result = None
    if request.method == 'POST':
        text = request.form.get('ad_text')
        lang = detect_language(text)
        issues = compliance_scan(translate_to_english(text, lang))
        result = translate_text(f"### Issues\n{issues}" if issues else "**Compliant!**", lang)
    return render_template('main.html', page='compliance', result=result)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        u = User(username=request.form['username'], password=request.form['password'], email=request.form['email'])
        db.session.add(u)
        db.session.commit()
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

