# main.py
import os
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from db import SessionLocal, init_db, User, Result, Feedback
from auth import get_password_hash, verify_password, create_access_token, decode_access_token
from model_utils import QUESTIONS, features_from_answers, fallback_score_from_answers, load_xgb_model, generate_roadmap_with_gemini
import json
import xgboost
init_db()
app = FastAPI(title="AI E-Learning Quiz API")

# serve frontend static files
# assumes project root structure: backend/ and frontend/ are siblings; adjust path if needed
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# load model if present
MODEL_PATH = os.getenv("XGB_MODEL_PATH", "./models/xgb_model.joblib")
MODEL, MODEL_TYPE = load_xgb_model(MODEL_PATH)

# Pydantic models
class RegisterPayload(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginPayload(BaseModel):
    email: EmailStr
    password: str

class SubmitPayload(BaseModel):
    answers: List[int]
    gemini_key: Optional[str] = None

class FeedbackPayload(BaseModel):
    result_id: Optional[int] = None
    rating: int
    comment: Optional[str] = None

# Helper: get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Auth helpers
def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    scheme, _, token = authorization.partition(" ")
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    user_id = int(payload.get("sub"))
    db = SessionLocal()
    user = db.query(User).get(user_id)
    db.close()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ---------- Auth endpoints ----------
@app.post("/api/register")
def register(payload: RegisterPayload, db=Depends(get_db)):
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(name=payload.name, email=payload.email, hashed_password=get_password_hash(payload.password))
    db.add(new_user); db.commit(); db.refresh(new_user)
    token = create_access_token({"sub": str(new_user.id), "email": new_user.email})
    return {"access_token": token, "token_type": "bearer", "user": {"id": new_user.id, "name": new_user.name, "email": new_user.email}}

@app.post("/api/login")
def login(payload: LoginPayload, db=Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": str(user.id), "email": user.email})
    return {"access_token": token, "token_type": "bearer", "user": {"id": user.id, "name": user.name, "email": user.email}}

# Return QUESTIONS so frontend can fetch if you'd like (index.html currently inlines it)
@app.get("/api/questions")
def get_questions():
    return QUESTIONS

# ---------- Submit quiz ----------
@app.post("/api/submit")
def submit(payload: SubmitPayload, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    if len(payload.answers) != len(QUESTIONS):
        raise HTTPException(status_code=400, detail=f"Expected {len(QUESTIONS)} answers")
    # model-based prediction (if model loaded)
    prediction_label = None
    model_score = None
    try:
        if MODEL is not None:
            X = features_from_answers(payload.answers)
            if MODEL_TYPE == "sklearn":
                pred = MODEL.predict(X)
                try:
                    proba = MODEL.predict_proba(X).tolist()
                except Exception:
                    proba = None
                prediction_label = str(pred[0])
                model_score = proba[0] if proba else None
            else:
                # xgboost booster
                if 'xgboost' in globals() and MODEL is not None:
                    dmat = xgboost.DMatrix(X)  # NOTE: only if xgboost imported
                    preds = MODEL.predict(dmat)
                    prediction_label = str(preds.tolist())
                    model_score = preds.tolist()
    except Exception as e:
        # ignore model errors, continue with fallback
        print("Model error:", e)
    # Always compute fallback (readable score)
    total_score, per_q = fallback_score_from_answers(payload.answers)
    roadmap = generate_roadmap_with_gemini(payload.gemini_key, current_user.name, total_score, per_q, prediction_label)
    res = Result(user_id=current_user.id, features=payload.answers, score=total_score, prediction=prediction_label, roadmap=roadmap)
    db.add(res); db.commit(); db.refresh(res)
    # compute correct indices (argmax of scores)
    correct_indices = [ q["scores"].index(max(q["scores"])) for q in QUESTIONS ]
    per_question_feedback = []
    for i, q in enumerate(QUESTIONS):
        chosen = int(payload.answers[i])
        per_question_feedback.append({
            "question": q["question"],
            "chosen_index": chosen,
            "chosen_text": q["options"][chosen],
            "is_correct": chosen == correct_indices[i],
            "correct_index": correct_indices[i],
            "correct_text": q["options"][correct_indices[i]],
            "score_for_choice": q["scores"][chosen]
        })
    return {
        "result_id": res.id,
        "total_score": total_score,
        "roadmap": roadmap,
        "per_question": per_question_feedback
    }

# ---------- Feedback endpoint ----------
@app.post("/api/feedback")
def submit_feedback(payload: FeedbackPayload, current_user: User = Depends(get_current_user), db=Depends(get_db)):
    fb = Feedback(user_id=current_user.id, result_id=payload.result_id, rating=payload.rating, comment=payload.comment)
    db.add(fb); db.commit(); db.refresh(fb)
    return {"status": "ok", "feedback_id": fb.id}

# ---------- Dashboard endpoint ----------
@app.get("/api/dashboard")
def dashboard(current_user: User = Depends(get_current_user), db=Depends(get_db)):
    results = db.query(Result).filter(Result.user_id == current_user.id).order_by(Result.created_at.desc()).all()
    feedbacks = db.query(Feedback).filter(Feedback.user_id == current_user.id).order_by(Feedback.created_at.desc()).all()
    return {
        "user": {"id": current_user.id, "name": current_user.name, "email": current_user.email},
        "results": [ {"id": r.id, "created_at": r.created_at.isoformat(), "score": r.score, "roadmap": r.roadmap} for r in results ],
        "feedbacks": [ {"id": f.id, "rating": f.rating, "comment": f.comment, "created_at": f.created_at.isoformat()} for f in feedbacks ]
    }

# Serve root page (frontend)
@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse("../frontend/index.html")
