import os
import json

FILE = "healthbackend/storage/users.json"


def _load_users():
    if not os.path.exists(FILE):
        return []
    with open(FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users):
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def check_credentials(username: str, password: str):
    """Check credentials and return user data if valid, None otherwise."""
    users = _load_users()
    for u in users:
        if u.get("username") == username and u.get("password") == password:
            return u
    return None


def create_user(username: str, password: str, full_name: str = "") -> bool:
    users = _load_users()
    if any(u.get("username") == username for u in users):
        return False
    users.append({"username": username, "password": password, "full_name": full_name})
    _save_users(users)
    return True
