class GroqRAGAgent:
    def __init__(self):
        # Chapters mapping
        self.chapters = {
            "1": "ROS 2 Basics",
            "2": "Python Agents & rclpy",
            "3": "URDF Humanoid Modeling",
            "4": "Simulation Techniques",
            "5": "AI Control Systems",
        }

        # Detailed chapter content (example, can expand)
        self.chapter_content = {
            "1": (
                "📘 **Chapter 1: ROS 2 Basics**\n\n"
                "ROS 2 (Robot Operating System 2) is a middleware framework used "
                "to build robotic applications. Key concepts:\n"
                "- Nodes: Small programs doing one task\n"
                "- Topics: Communication channels\n"
                "- Publishers/Subscribers: Send & receive data\n"
                "- Services: Request–response communication\n"
                "- DDS: Fast & real-time communication system"
            ),
            "2": (
                "📘 **Chapter 2: Python Agents & rclpy**\n\n"
                "Python agents in ROS 2 use the `rclpy` library to create nodes, "
                "publish/subscribe to topics, and call services. "
                "You can write autonomous robot behaviors using Python scripts."
            ),
            "3": (
                "📘 **Chapter 3: URDF Humanoid Modeling**\n\n"
                "URDF (Unified Robot Description Format) is used to define a robot's "
                "physical model (links, joints, sensors). Humanoid robots are modeled "
                "using URDF for simulation and control."
            ),
            "4": (
                "📘 **Chapter 4: Simulation Techniques**\n\n"
                "Simulation environments like Gazebo or Webots allow testing of robots "
                "without physical hardware. You can simulate sensors, motors, and physics."
            ),
            "5": (
                "📘 **Chapter 5: AI Control Systems**\n\n"
                "AI control systems integrate planning, perception, and actuation. "
                "They use sensors and algorithms to make autonomous decisions in robots."
            ),
        }

    def answer(self, query: str):
        q = query.lower().strip()

        # Greeting only for hi/hello/hey
        if q in ["hi", "hello", "hey"]:
            return (
                "Hi 👋 How can I help you?\n\n"
                "This textbook contains the following chapters:\n"
                "1. ROS 2 Basics\n"
                "2. Python Agents & rclpy\n"
                "3. URDF Humanoid Modeling\n"
                "4. Simulation Techniques\n"
                "5. AI Control Systems\n\n"
                "You can ask questions from these chapters and I’ll help you 😊"
            )

        # Check for chapter keywords
        for chap_num, chap_name in self.chapters.items():
            if chap_num in q or chap_name.lower() in q:
                return self.chapter_content[chap_num]

        # Fallback
        return "Please ask a question related to the textbook chapters 😊"
