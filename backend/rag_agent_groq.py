class GroqRAGAgent:
    def __init__(self):
        self.chapters = {
            "1": "ROS 2 Basics",
            "2": "Python Agents & rclpy",
            "3": "URDF Humanoid Modeling",
            "4": "Simulation Techniques",
            "5": "AI Control Systems",
        }

    def answer(self, query: str):
        q = query.lower().strip()

        # 1️⃣ Greeting only once
        if q in ["hi", "hello", "hey"]:
            return {
                "answer": (
                    "Hi 👋 How can I help you?\n\n"
                    "This textbook contains the following chapters:\n"
                    "1. ROS 2 Basics\n"
                    "2. Python Agents & rclpy\n"
                    "3. URDF Humanoid Modeling\n"
                    "4. Simulation Techniques\n"
                    "5. AI Control Systems\n\n"
                    "You can ask questions from these chapters and I’ll help you 😊"
                ),
                "status": "success",
            }

        # 2️⃣ Chapter 1 handling
        if "chapter 1" in q or "chap 1" in q or "ros" in q:
            return {
                "answer": (
                    "📘 **Chapter 1: ROS 2 Basics**\n\n"
                    "ROS 2 (Robot Operating System 2) is a middleware framework used "
                    "to build robotic applications. It allows different parts of a robot "
                    "like sensors, motors, and AI modules to communicate using nodes.\n\n"
                    "Key concepts:\n"
                    "- Nodes: Small programs doing one task\n"
                    "- Topics: Communication channels\n"
                    "- Publishers/Subscribers: Send & receive data\n"
                    "- Services: Request–response communication\n"
                    "- DDS: Underlying fast & real-time communication system\n\n"
                    "ROS 2 is widely used in drones, self-driving cars, humanoid robots, "
                    "and industrial automation."
                ),
                "status": "success",
                "confidence": "high",
            }

        # 3️⃣ Default fallback
        return {
            "answer": "Please ask a question related to the textbook chapters 😊",
            "status": "empty",
        }
