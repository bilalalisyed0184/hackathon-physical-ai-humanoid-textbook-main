class GroqRAGAgent:
    def __init__(self):
        # Chapters dictionary
        self.chapters = {
            "1": {
                "title": "ROS 2 Basics",
                "summary": "📘 **Chapter 1: ROS 2 Basics** ROS 2 is a middleware framework used to build robotic applications. Nodes communicate via topics, services, and DDS."
            },
            "2": {
                "title": "Python Agents & rclpy",
                "summary": "📘 **Chapter 2: Python Agents & rclpy** Python agents in ROS 2 use rclpy to create nodes, publish/subscribe to topics, and call services."
            },
            "3": {
                "title": "URDF Humanoid Modeling",
                "summary": "📘 **Chapter 3: URDF Humanoid Modeling** URDF allows defining robot models in XML, including joints, links, sensors, and actuators."
            },
            "4": {
                "title": "Simulation Techniques",
                "summary": "📘 **Chapter 4: Simulation Techniques** Use Gazebo or other simulators to test robot behaviors in virtual environments."
            },
            "5": {
                "title": "AI Control Systems",
                "summary": "📘 **Chapter 5: AI Control Systems** Covers control algorithms, AI-based decision making, and autonomous behaviors."
            }
        }

    def answer(self, query: str) -> str:
        query_lower = query.lower().strip()

        # 1️⃣ Fallback greeting
        greetings = ["hi", "hello", "hey"]
        if query_lower in greetings:
            return (
                "Hi 👋 How can I help you?\n"
                "This textbook contains the following chapters:\n"
                "1. ROS 2 Basics\n"
                "2. Python Agents & rclpy\n"
                "3. URDF Humanoid Modeling\n"
                "4. Simulation Techniques\n"
                "5. AI Control Systems\n"
                "You can ask questions from these chapters and I’ll help you 😊"
            )

        # 2️⃣ Chapter queries
        for num, chapter in self.chapters.items():
            if num in query_lower or chapter["title"].lower() in query_lower:
                return chapter["summary"]

        # 3️⃣ Free text keywords (optional basic detection)
        keywords = {
            "ros": "📘 **Chapter 1: ROS 2 Basics** Nodes, Topics, Publishers/Subscribers, Services, DDS.",
            "python agent": self.chapters["2"]["summary"],
            "urdf": self.chapters["3"]["summary"],
            "simulation": self.chapters["4"]["summary"],
            "ai control": self.chapters["5"]["summary"],
        }

        for key, ans in keywords.items():
            if key in query_lower:
                return ans

        # 4️⃣ Fallback message
        return "Please ask a question related to the textbook chapters 😊"
