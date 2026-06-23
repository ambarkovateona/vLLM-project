from database import init_db, register_user

# Initialize database and tables
init_db()
print("Database initialized.")

# Create test users
test_users = [
    ("admin",  "admin123"),
    ("user1",  "pass123"),
    ("user2",  "pass456"),
]

for username, password in test_users:
    if register_user(username, password):
        print(f"Created user: {username}")
    else:
        print(f"User already exists: {username}")

print("\nDone. Run app.py to start the application.")