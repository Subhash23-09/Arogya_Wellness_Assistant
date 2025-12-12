# Import necessary modules for file handling and JSON serialization
import json
import os

# Base dir: .../healthbackend
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
FILE = os.path.join(STORAGE_DIR, "history.json")


def load():
    """Read and return the stored data from the history file."""
    if not os.path.exists(FILE):
        return {}
    with open(FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save(data: dict):
    """Persist the given history dict to the JSON file."""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    with open(FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



# ------------------------------------------------------------
# Save user history entry
# ------------------------------------------------------------
# Purpose: Append a new history entry for a specific user.
# If the user does not exist, initialize their entry as an empty list.
def save_history(user_id, entry):
    # Load the existing history structure
    data = load()
    # Create a list for the user if not present and append the new entry
    data.setdefault(user_id, []).append(entry)
    # Save updated data back to storage
    save(data)


# ------------------------------------------------------------
# Retrieve user history
# ------------------------------------------------------------
# Purpose: Fetch the conversation or history list for a given user.
# Returns an empty list if the user has no recorded history.
def get_history(user_id):
    return load().get(user_id, [])

# ------------------------------------------------------------
# Save user history entry
# ------------------------------------------------------------
# Purpose: Append a new history entry for a specific user.
# If the user does not exist, initialize their entry as an empty list.
def save_history(user_id, entry):
    # Load the existing history structure
    data = load()
    # Create a list for the user if not present and append the new entry
    data.setdefault(user_id, []).append(entry)
    # Save updated data back to storage
    save(data)


# ------------------------------------------------------------
# Retrieve user history
# ------------------------------------------------------------
# Purpose: Fetch the conversation or history list for a given user.
# Returns an empty list if the user has no recorded history.
def get_history(user_id):
    return load().get(user_id, [])
