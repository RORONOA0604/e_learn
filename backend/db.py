# db.py
import os
from sqlalchemy import create_engine, Column, Integer, String, JSON, Float, DateTime, ForeignKey, Text, func
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_elearn.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    results = relationship("Result", back_populates="user")
    feedbacks = relationship("Feedback", back_populates="user")

class Result(Base):
    __tablename__ = "results"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    features = Column(JSON)
    score = Column(Float)
    prediction = Column(String)
    roadmap = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="results")

class Feedback(Base):
    __tablename__ = "feedbacks"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    result_id = Column(Integer, ForeignKey("results.id"), nullable=True)
    rating = Column(Integer)  # 1..5
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship("User", back_populates="feedbacks")

def init_db():
    Base.metadata.create_all(bind=engine)
