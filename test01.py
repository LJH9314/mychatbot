# Install required libraries
import random
import openai
import json
import time
import os
import streamlit as st
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Test connection


def test_connection(client):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "Hello! Please respond with 'Connection successful!'"}],
            max_tokens=10
        )
        print("✅Connected!", response.choices[0].message.content)
        #st.text("✅Connected!" + response.choices[0].message.content)
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        st.text(f"❌ Connection failed: {e}")
        return False


class NodeType(Enum):
    """Enumeration of node types in the state graph."""
    INTERACTIVE = 1    # Interactive nodes that ask questions or perform actions
    DIRECT = 2         # Direct response nodes that provide information
    FUNCTION = 3       # Function nodes that execute specific functions


@dataclass
class BotConfiguration:
    """Configuration for bot behavior and personality."""
    name: str
    personality: str
    context: str
    extraction_fields: List[str]
    custom_functions: Dict[str, callable]
    system_prompt: str
    max_tokens: int = 200
    temperature: float = 0.1


class StateNode:
    """Represents a single node in the state machine."""

    def __init__(self, node_data: Dict[str, Any]):
        self.id = node_data["id"]
        self.type = NodeType(node_data["type"])
        self.text = node_data.get("text", "")
        self.function = node_data.get("function", None)
        self.transition_list = node_data.get("transition_list", [])

    def __repr__(self):
        return f"StateNode(id={self.id}, type={self.type.name}, text='{self.text[:50]}...')"


class DynamicStateGraph:
    """Universal state machine graph manager."""

    def __init__(self, json_data: List[Dict], scenario_name: str = "Unknown"):
        self.nodes = {}
        self.current_node_id = 0
        self.scenario_name = scenario_name
        self.load_from_data(json_data)

    def load_from_data(self, json_data: List[Dict]):
        """Load state graph from JSON data."""
        try:
            for node_data in json_data:
                node = StateNode(node_data)
                self.nodes[node.id] = node

            print(
                f"📊 Loaded {len(self.nodes)} nodes for '{self.scenario_name}' scenario")

        except Exception as e:
            print(f"❌ Error loading state graph: {e}")
            raise

    @classmethod
    def from_file(cls, json_file_path: str, scenario_name: str = None):
        if scenario_name is None:
            scenario_name = os.path.basename(
                json_file_path).replace('.json', '')

        with open(json_file_path, "r") as f:  # 使用传入的参数路径
            json_data = json.load(f)

        return cls(json_data, scenario_name)

    def get_current_node(self) -> StateNode:
        """Get the current node in the state machine."""
        return self.nodes[self.current_node_id]

    def get_node(self, node_id: int) -> Optional[StateNode]:
        """Get a specific node by ID."""
        return self.nodes.get(node_id)

    def transition_to(self, node_id: int) -> bool:
        """Transition to a specific node if it exists."""
        if node_id in self.nodes:
            self.current_node_id = node_id
            return True
        return False

    def reset(self):
        """Reset to the initial node."""
        self.current_node_id = 0


class DynamicChatEngine:
    """Advanced chat engine that works with any state graph and bot configuration."""

    def __init__(self, state_graph: DynamicStateGraph, bot_config: BotConfiguration, client):
        self.state_graph = state_graph
        self.bot_config = bot_config
        self.client = client
        self.conversation_history = []
        self.extracted_data = {}
        self.is_running = True
        self.session_started = False

        # Initialize extracted data fields
        for field in bot_config.extraction_fields:
            self.extracted_data[field] = None

    def start_conversation(self) -> str:
        """Start the conversation with the initial node."""
        self.session_started = True
        self.state_graph.reset()
        current_node = self.state_graph.get_current_node()

        if current_node.type == NodeType.DIRECT:
            response = current_node.text
        else:
            response = self._generate_response(current_node.text, "")

        self.conversation_history.append(
            {"role": "assistant", "content": response})
        return response

    def process_user_input(self, user_input: str) -> str:
        """Process user input and generate appropriate response."""
        if not self.session_started:
            return self.start_conversation()

        # Add user input to conversation history
        self.conversation_history.append(
            {"role": "user", "content": user_input})

        # Extract data from user input
        self._extract_data(user_input)

        # Get current node
        current_node = self.state_graph.get_current_node()

        # Determine next node based on transitions
        next_node_id = self._determine_next_node(current_node, user_input)

        if next_node_id is not None:
            # Transition to next node
            if self.state_graph.transition_to(next_node_id):
                next_node = self.state_graph.get_current_node()
                response = self._handle_node(next_node, user_input)
            else:
                response = "I'm sorry, I encountered an error. Let me help you differently."
        else:
            # Stay on current node and ask for clarification
            response = self._generate_clarification_response(
                current_node, user_input)

        self.conversation_history.append(
            {"role": "assistant", "content": response})
        return response

    def _handle_node(self, node: StateNode, user_input: str) -> str:
        """Handle different types of nodes."""
        if node.type == NodeType.DIRECT:
            return node.text
        elif node.type == NodeType.INTERACTIVE:
            return self._generate_response(node.text, user_input)
        elif node.type == NodeType.FUNCTION:
            return self._execute_function(node, user_input)
        else:
            return "I'm not sure how to handle this request."

    def _generate_response(self, node_text: str, user_input: str) -> str:
        """Generate AI response based on node text and user input."""
        messages = [
            {"role": "system", "content": self.bot_config.system_prompt},
            {"role": "system", "content": f"Current task: {node_text}"},
            {"role": "system", "content": f"Context: {self.bot_config.context}"},
            {"role": "system", "content": f"Extracted data so far: {self.extracted_data}"}
        ]

        # Add recent conversation history
        messages.extend(self.conversation_history[-3:])

        if user_input:
            messages.append({"role": "user", "content": user_input})

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=self.bot_config.max_tokens,
                temperature=self.bot_config.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"I apologize, I'm having trouble processing your request. Error: {str(e)}"

    def _determine_next_node(self, current_node: StateNode, user_input: str) -> Optional[int]:
        """Determine the next node based on user input and transition conditions."""
        if not current_node.transition_list:
            return None

        # Use AI to determine which transition condition matches
        transition_options = []
        for i, transition in enumerate(current_node.transition_list):
            transition_options.append(
                f"{i}: {transition['condition_text']} -> Node {transition['next_node_id']}")

        decision_prompt = f"""
        Given the user input: "{user_input}"
        And the extracted data: {self.extracted_data}

        Which of these transition conditions best matches? Return only the number (0, 1, 2, etc.):
        {chr(10).join(transition_options)}

        If none match well, return -1.
        """

        try:
            decision_response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a state transition decision maker. Return only the number of the best matching condition."},
                    {"role": "user", "content": decision_prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )

            decision = decision_response.choices[0].message.content.strip()
            try:
                choice_idx = int(decision)
                if 0 <= choice_idx < len(current_node.transition_list):
                    return current_node.transition_list[choice_idx]['next_node_id']
            except ValueError:
                pass
        except Exception:
            pass

        return None

    def _extract_data(self, user_input: str):
        """Extract relevant data from user input."""
        for field in self.bot_config.extraction_fields:
            # Only extract if not already extracted
            if self.extracted_data[field] is None:
                extraction_prompt = f"""
                Extract the {field} from this user input: "{user_input}"
                If the {field} is clearly mentioned, return just the value.
                If not clearly mentioned, return "NOT_FOUND".
                """

                try:
                    response = self.client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "You are a data extraction assistant. Return only the extracted value or 'NOT_FOUND'."},
                            {"role": "user", "content": extraction_prompt}
                        ],
                        max_tokens=50,
                        temperature=0.1
                    )

                    extracted_value = response.choices[0].message.content.strip(
                    )
                    if extracted_value != "NOT_FOUND":
                        self.extracted_data[field] = extracted_value
                except Exception:
                    pass

    def _execute_function(self, node: StateNode, user_input: str) -> str:
        """Execute a function node."""
        if node.function:
            # Handle the case where node.function is a dictionary with 'name' field
            if isinstance(node.function, dict):
                function_name = node.function.get('name')
            else:
                # Handle the case where node.function is a string
                function_name = node.function

            # Check if the function exists in custom_functions
            if function_name and function_name in self.bot_config.custom_functions:
                try:
                    return self.bot_config.custom_functions[function_name](self.extracted_data, user_input)
                except Exception as e:
                    return f"Function execution failed: {str(e)}"
            else:
                # Function not found in custom_functions, simulate it
                return f"Simulated function execution: {function_name or str(node.function)}"
        else:
            return "No function specified for this node."

    def _generate_clarification_response(self, node: StateNode, user_input: str) -> str:
        """Generate a clarification response when no transition matches."""
        clarification_prompt = f"""
        The user said: "{user_input}"
        Current task: {node.text}

        Generate a helpful clarification question or response to keep the conversation moving forward.
        Be natural and conversational, matching the personality: {self.bot_config.personality}
        """

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self.bot_config.system_prompt},
                    {"role": "user", "content": clarification_prompt}
                ],
                max_tokens=self.bot_config.max_tokens,
                temperature=self.bot_config.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "I'm not sure I understood. Could you please clarify?"

    def reset_session(self):
        """Reset the chat session."""
        self.conversation_history = []
        self.extracted_data = {
            field: None for field in self.bot_config.extraction_fields}
        self.is_running = True
        self.session_started = False
        self.state_graph.reset()


# 添加航班助手配置
# Custom functions for Flight System
def book_flight(extracted_data: dict, user_input: str) -> str:
    """Simulated function to book a flight"""
    name = extracted_data.get('passenger_name', 'customer')
    origin = extracted_data.get('origin', 'unknown city')
    destination = extracted_data.get('destination', 'unknown city')
    date = extracted_data.get('date', 'unknown date')
    return f"Flight booked for {name} from {origin} to {destination} on {date}. Confirmation code: FLT{int(time.time()) % 10000}"


def check_flight_status(extracted_data: dict, user_input: str) -> str:
    """Simulated function to check flight status"""
    flight_num = extracted_data.get('flight_number', 'unknown').upper()
    statuses = ["on time", "delayed by 30 mins", "boarding", "cancelled"]
    return f"Flight {flight_num} is currently {random.choice(statuses)}. Gate: {random.randint(1, 50)}"


def change_flight(extracted_data: dict, user_input: str) -> str:
    """Simulated function to change flight"""
    old_flight = extracted_data.get('flight_number', 'unknown')
    new_date = extracted_data.get('new_date', 'unknown')
    return f"Your flight {old_flight} has been changed to {new_date}. New confirmation code: CHG{int(time.time()) % 10000}"


def cancel_flight(extracted_data: dict, user_input: str) -> str:
    """Simulated function to cancel flight"""
    flight_num = extracted_data.get('flight_number', 'unknown')
    return f"Flight {flight_num} has been cancelled. Refund reference: REF{int(time.time()) % 10000}"


def transfer_to_agent(extracted_data: dict, user_input: str) -> str:
    """Simulated function to transfer to human agent"""
    return "Transferring you to a human agent. Please wait while we connect you..."


def analyze_flight_scenario():
    """Analyze the flight assistant scenario."""
    print("\n📊 FLIGHT ASSISTANT ANALYSIS")
    print("=" * 60)

    # Basic statistics
    flight_nodes = len(flight_state_graph.nodes)
    print(f"\n📈 System Scale: {flight_nodes} nodes")

    # Node type analysis
    type_counts = {1: 0, 2: 0, 3: 0}
    for node in flight_state_graph.nodes.values():
        type_counts[node.type.value] += 1

    print(f"\n🔍 Node Type Distribution:")
    print(
        f"  Interactive (Type 1): {type_counts[1]} - Questions and user interactions")
    print(f"  Direct (Type 2):      {type_counts[2]} - Information responses")
    print(f"  Function (Type 3):    {type_counts[3]} - System operations")

    # Configuration analysis
    print(f"\n⚙️ Configuration Details:")
    print(f"Extraction Fields: {len(flight_config.extraction_fields)}")
    print(f"Custom Functions:  {len(flight_config.custom_functions)}")

    # Key features
    print(f"\n🛠️ System Capabilities:")
    print("- Flight booking and management")
    print("- Real-time flight status checks")
    print("- Flight changes and cancellations")
    print("- Travel planning assistance")
    print("- Seamless agent transfer")

    print("\n✅ Analysis complete!")


def export_flight_conversation(filename: str = "flight_conversation.json"):
    """Export the current flight conversation to a file."""
    try:
        with open(filename, 'w') as f:
            json.dump({
                'scenario': flight_state_graph.scenario_name,
                'conversation': flight_chat_engine.conversation_history,
                'extracted_data': flight_chat_engine.extracted_data,
                'current_node': flight_chat_engine.state_graph.current_node_id
            }, f, indent=2)
        print(f"💾 Flight conversation exported to {filename}")
    except Exception as e:
        print(f"❌ Export failed: {e}")


def switch_scenario(scenario_name: str):
    """Switch between different scenarios (only flight available)."""
    if scenario_name.lower() == 'flight':
        flight_chat_engine.reset_session()
        print("✈️ Switched to Flight Assistant System")
        return flight_chat_engine.start_conversation()
    else:
        print("❌ Unknown scenario. Currently only 'flight' is available")
        return None


def main_page_setting():
    # Configuring Streamlit page settings
    st.set_page_config(page_title="Flight Assistant Chat",
                       page_icon="💬", layout="wide")

    st.markdown(
        """
        <style>
        .chat-container {
            max-height: 60vh;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Main chat interface
    st.title("🤖 Flight Assistant - ChatBot")

def main():  
    
    # Configuring Streamlit page settings
    main_page_setting()
    
    # chat engine status
    if "chatengine_init" not in st.session_state:
        st.session_state.chatengine_init = False
    
    # Initialize chat session in Streamlit if not already present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "system",
                "content": "You are a super useful assistant.",
            },
        ]
    if not st.session_state.chatengine_init:    # chat engine does't init
        # Set up the DeepSeek API client
        API_KEY = r"sk-97254a1fb02949ff9eff466bfe343b72"
        client = openai.OpenAI(
            api_key=API_KEY,
            base_url="https://api.deepseek.com"
        )
        # Test connection
        if not test_connection(client):
            st.session_state.chatengine_init = False
            st.rerun()  # 重新运行应用
        else:
            st.text("✅Connected! Flight Assistant is ready!")
        # 航班助手配置
        # Flight Assistant Configuration
        flight_config = BotConfiguration(
            name="Flight Assistant",
            personality="Professional, helpful and efficient travel assistant",
            context="Airline booking system for flight reservations and management",
            extraction_fields=[
                "passenger_name",
                "flight_number",
                "origin",
                "destination",
                "date",
                "new_date",
                "seat_preference"
            ],
            custom_functions={
                "book_flight": book_flight,
                "check_flight_status": check_flight_status,
                "change_flight": change_flight,
                "cancel_flight": cancel_flight,
                "transfer_to_agent": transfer_to_agent,
                "check_appt_detail": check_flight_status,  # 兼容旧节点
                "transfer_call": transfer_to_agent        # 兼容旧节点
            },
            system_prompt="""You are a professional flight booking assistant. Help customers with:
            - Booking new flights
            - Checking flight status
            - Changing/cancelling reservations
            - Travel planning
        Be clear, efficient and always confirm important details.""",
            max_tokens=250,
            temperature=0.2
        )

        # 加载航班状态图 (替换原来的医疗系统加载)
        flight_state_graph = DynamicStateGraph.from_file(
            r'C:\Users\LiuJH\Desktop\Final\flight_data.json', 'Flight Assistant System')

        # 初始化航班聊天引擎 (替换原来的医疗引擎)        
        flight_chat_engine = DynamicChatEngine(flight_state_graph, flight_config, client)

        if "flight_chat_engine" not in st.session_state:
            st.session_state.flight_chat_engine = flight_chat_engine
        
        print("FLIGHT ASSISTANT SYSTEM")
        print("\n" + "="*60)

        # 启动对话
        #initial_message = flight_chat_engine.start_conversation()
        initial_message = st.session_state.flight_chat_engine.start_conversation()
        print(f"Assistant: {initial_message}")

        # init msg append
        st.session_state.chat_history.append(
            {"role": "assistant", "content": initial_message})

        # Interactive flight chat session
        print("\n✈️ Flight Assistant Session Started")
        # print("Type 'quit' to end the conversation\n")

        #st.text("\n✈️ Flight Assistant Session Started")
        st.button("Clear Chat", on_click=lambda: st.session_state.pop(
            "chat_history", None))

        # Create a container for the scrollable chat history
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        chat_container = st.container()
        st.markdown("</div>", unsafe_allow_html=True)

        # Create a container for the fixed input field at the bottom
        input_container = st.container()

        # Display chat history in the scrollable container
        with chat_container:
            for message in st.session_state.chat_history[1:]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Input field for user's message (fixed at the bottom)
        with input_container:
            user_prompt = st.chat_input("Ask...", key="user_input")

        st.session_state.chatengine_init = True
    else:        
        # st container start
        st.text("✅Connected! Flight Assistant is ready!")

        st.button("Clear Chat", on_click=lambda: st.session_state.pop(
            "chat_history", None))
        
        # Create a container for the scrollable chat history
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        chat_container = st.container()
        st.markdown("</div>", unsafe_allow_html=True)

        # Create a container for the fixed input field at the bottom
        input_container = st.container()

        # Display chat history in the scrollable container
        with chat_container:
            for message in st.session_state.chat_history[1:]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Input field for user's message (fixed at the bottom)
        with input_container:
            user_prompt = st.chat_input("Ask...", key="user_input")

        if user_prompt:
            # Add user's message to chat and display it
            with chat_container:
                st.chat_message("user").markdown(user_prompt)
            st.session_state.chat_history.append(
                {"role": "user", "content": user_prompt})

            # Generate a response and measure the time it takes
            start_time = time.time()
            user_input = user_prompt  # user input
            if user_input:
                response = st.session_state.flight_chat_engine.process_user_input(
                    user_input)  # model response
                print(f"Assistant: {response}")
            end_time = time.time()

            # Calculate the time taken for response generation
            generation_time = end_time - start_time
            print(f"Response generated in {generation_time:.2f} seconds")

            # Extract the assistant's response correctly
            assistant_response = response
            st.session_state.chat_history.append(
                {"role": "assistant", "content": assistant_response}
            )

            # Display response
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(
                        assistant_response + f"  *Response generated in {generation_time:.2f} seconds.*")
        # a st container end
 
if __name__ == "__main__":
    main()
