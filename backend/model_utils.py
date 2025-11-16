# model_utils.py
import os
import json
import joblib
import numpy as np
import requests

try:
    import xgboost as xgb
except Exception:
    xgb = None

QUESTIONS = [
    { "question": "Which of the following best describes a 'for' loop?",
      "options": ["A conditional statement", "A way to store multiple values", "A control flow statement for iterating", "A function that calls itself"],
      "scores": [2,3,10,1]},
    { "question": "How would you efficiently create a new list of only even numbers from an existing list?",
      "options": ["Series of 'if-else' statements", "A loop that checks each number", "A function for every possible number", "Store numbers in separate variables"],
      "scores": [3,10,1,2]},
    { "question": "What is the primary purpose of a function in programming?",
      "options": ["To stop the program", "To store data", "To group reusable code", "To create comments"],
      "scores": [1,3,10,2]},
    { "question": "If a coin is flipped twice, what is the probability of getting two heads?",
      "options": ["1/2", "1/3", "1/4", "1"],
      "scores": [3,2,10,1]},
    { "question": "What does the 'mean' of a dataset represent?",
      "options": ["The middle value", "The most frequent value", "The average of all values", "The range of values"],
      "scores": [4,3,10,2]},
    { "question": "An upward trend in a graph where both variables increase together is called:",
      "options": ["Negative correlation", "Positive correlation", "No correlation", "A statistical error"],
      "scores": [1,10,2,1]},
    { "question": "What is the role of a server in a client-server relationship?",
      "options": ["To request information", "To provide resources or services", "The user's main computer", "To protect from viruses"],
      "scores": [3,10,2,4]},
    { "question": "Which of the following best describes 'the cloud'?",
      "options": ["A single, massive computer", "A network of servers accessed via internet", "A personal storage device", "Software for documents"],
      "scores": [2,10,1,1]},
    { "question": "What is an API (Application Programming Interface)?",
      "options": ["A visual user interface", "A set of rules for app communication", "A database for storage", "A security protocol"],
      "scores": [3,10,2,4]},
    { "question": "When designing a website, what is the most important consideration?",
      "options": ["Using many colors", "Intuitive user navigation", "Complex animations", "Smallest font size"],
      "scores": [2,10,1,1]},
    { "question": "What's the best way to create visual hierarchy on a poster?",
      "options": ["Make all text the same", "Make important info larger/bolder", "Put least important info at top", "Fill every empty space"],
      "scores": [1,10,1,2]},
    { "question": "What is the primary goal of UX (User Experience) design?",
      "options": ["Making a product look appealing", "Focusing on branding/logo", "Enhancing usability and accessibility", "Writing the application code"],
      "scores": [4,2,10,1]},
    { "question": "What's a constructive way to handle disagreement with a teammate?",
      "options": ["Ignore their idea", "Publicly criticize them", "Discuss pros and cons together", "Complain to someone else"],
      "scores": [1,1,10,1]},
    { "question": "What is the main purpose of version control software like Git?",
      "options": ["To write code automatically", "To track changes and manage collaboration", "To design the UI", "To store only the final version"],
      "scores": [1,10,1,2]},
    { "question": "When presenting to stakeholders, it's most important to:",
      "options": ["Use highly technical jargon", "Focus only on what went wrong", "Clearly communicate progress and challenges", "Make the presentation as long as possible"],
      "scores": [1,1,10,1]}
]

def load_xgb_model(path="models/xgb_model.joblib"):
    if not os.path.exists(path):
        alt = os.getenv("XGB_MODEL_PATH", path)
        if os.path.exists(alt):
            path = alt
        else:
            return None, None
    # attempt joblib load
    try:
        m = joblib.load(path)
        return m, "sklearn"
    except Exception:
        # try native xgboost model
        if xgb is not None:
            try:
                booster = xgb.Booster()
                booster.load_model(path)
                return booster, "booster"
            except Exception:
                return None, None
        return None, None

def features_from_answers(answers):
    n = len(QUESTIONS)
    if len(answers) != n:
        raise ValueError(f"Expected {n} answers, got {len(answers)}")
    return np.array([answers], dtype=np.float32)

def fallback_score_from_answers(answers):
    total = 0
    per_q = []
    for q, sel in zip(QUESTIONS, answers):
        sc = q["scores"][int(sel)]
        total += sc
        per_q.append({"question": q["question"], "selected": q["options"][int(sel)], "score": sc})
    return total, per_q

def generate_roadmap_with_gemini(api_key: str, user_name: str, total_score: float, per_question: list, prediction: str=None):
    strengths = []
    weaknesses = []
    for item in per_question:
        if item["score"] >= 8:
            strengths.append(item["question"])
        elif item["score"] <= 3:
            weaknesses.append(item["question"])

    prompt = (
        f"Create a personalized 6-week learning roadmap for {user_name} in JSON.\n"
        f"Total score: {total_score}\n"
        f"Strengths: {strengths}\n"
        f"Weaknesses: {weaknesses}\n\n"
        "Output a JSON object with keys: summary (string), weeks (array of {week, goal, activities}), resources (array of {title, url, why})."
    )

    gemini_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        # fallback heuristic plan
        return {
            "summary": "No Gemini key — heuristic plan provided.",
            "weeks": [
                {"week": 1, "goal": "Basics refresh", "activities": ["Revise loops & functions", "Small coding tasks"]},
                {"week": 2, "goal": "Data structures", "activities": ["Practice lists, dicts, comprehensions"]},
                {"week": 3, "goal": "Probability & statistics basics", "activities": ["Mean, probability tasks"]},
                {"week": 4, "goal": "APIs & Systems", "activities": ["Build a small API endpoint"]},
                {"week": 5, "goal": "Design & UX", "activities": ["Design a poster, apply visual hierarchy"]},
                {"week": 6, "goal": "Collaboration & presentation", "activities": ["Use Git, prepare a presentation"]},
            ],
            "resources": [
                {"title": "LeetCode", "url": "https://leetcode.com", "why": "Short focused problems"},
                {"title": "MDN Web Docs", "url": "https://developer.mozilla.org", "why": "Web fundamentals"},
            ],
        }

    # Placeholder endpoint — replace with real Gemini/Vertex AI call if needed.
    endpoint = "https://api.generativeai.example/v1/generate"  # <- CHANGE THIS
    headers = {"Authorization": f"Bearer {gemini_api_key}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "max_output_tokens": 800, "temperature": 0.2}

    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=20)
        r.raise_for_status()
        resp = r.json()
        # adapt depending on real response shape
        text = resp.get("output_text") or resp.get("result") or json.dumps(resp)
        try:
            return json.loads(text)
        except Exception:
            return {"summary": text}
    except Exception as e:
        return {"summary": f"Gemini call failed: {str(e)}. Returning heuristic roadmap."}
