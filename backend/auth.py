# auth.py
import os
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("JWT_SECRET", "dev_secret_change_me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# bcrypt truncation limit (bytes)
BCRYPT_MAX_BYTES = 72

def _truncate_password_for_bcrypt(password: str) -> str:
    """
    Truncate a string so its UTF-8 encoding is at most 72 bytes,
    returning a valid UTF-8 string (drops any incomplete trailing byte sequences).
    This ensures deterministic hashing/verifying behavior when using bcrypt.
    """
    if password is None:
        return ""
    b = password.encode('utf-8', errors='ignore')
    if len(b) <= BCRYPT_MAX_BYTES:
        return password
    # truncate bytes and decode ignoring incomplete sequences
    truncated = b[:BCRYPT_MAX_BYTES].decode('utf-8', errors='ignore')
    return truncated

def verify_password(plain: str, hashed: str) -> bool:
    # truncate plain before verifying to match how we hashed it
    p = _truncate_password_for_bcrypt(plain)
    return pwd_ctx.verify(p, hashed)

def get_password_hash(password: str) -> str:
    p = _truncate_password_for_bcrypt(password)
    return pwd_ctx.hash(p)

def create_access_token(data: dict, expires_delta: int = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=(expires_delta or ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
