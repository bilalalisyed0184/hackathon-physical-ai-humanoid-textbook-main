TEXTBOOK_CHAPTERS = [
    "1. ROS 2 Basics",
    "2. Python Agents & rclpy",
    "3. URDF Humanoid Modeling",
    "4. Simulation Techniques",
    "5. AI Control Systems",
]

def default_greeting():
    chapters = "\n".join(TEXTBOOK_CHAPTERS)
    return (
        "Hi 👋 How can I help you?\n\n"
        "This textbook contains the following chapters:\n\n"
        f"{chapters}\n\n"
        "You can ask me questions from these chapters and I’ll help you 😊"
    )
