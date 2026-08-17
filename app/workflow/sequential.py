import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END


# Load environment variables
load_dotenv()

# --------------------------------------------------
# 1. Define the State
# --------------------------------------------------

class PipelineState(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str


# --------------------------------------------------
# 2. Initialize the LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)


# --------------------------------------------------
# 3. Editor Node
# --------------------------------------------------

def editor_node(state: PipelineState) -> dict:
    """
    Stage 1:
    Cleans grammar, spelling mistakes, and improves readability.
    """

    print("\n--- [Stage 1] Executing Editor Node ---")

    prompt = (
        "You are an expert copyeditor. Clean up the following raw text. "
        "Fix grammatical errors, spelling mistakes, and improve the flow "
        "while keeping the original meaning and core message intact. "
        "Do not add unnecessary information. "
        "Return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )

    response = llm.invoke(prompt)

    return {
        "edited_text": response.content.strip()
    }


# --------------------------------------------------
# 4. Scriptwriter Node
# --------------------------------------------------

def scriptwriter_node(state: PipelineState) -> dict:
    """
    Stage 2:
    Converts the edited text into an engaging video script.
    """

    print("\n--- [Stage 2] Executing Scriptwriter Node ---")

    prompt = (
        "You are a charismatic YouTube content creator. "
        "Take the following edited text and transform it into "
        "an engaging, punchy, conversational video script. "
        "Make it sound like a real person speaking naturally "
        "and passionately. "
        "Keep the original information and meaning intact. "
        "Return only the script content.\n\n"
        f"Edited Text:\n{state['edited_text']}"
    )

    response = llm.invoke(prompt)

    return {
        "script_text": response.content.strip()
    }


# --------------------------------------------------
# 5. Hindi Translator Node
# --------------------------------------------------

def translator_node(state: PipelineState) -> dict:
    """
    Stage 3:
    Translates the script into natural conversational Hindi.
    """

    print("\n--- [Stage 3] Executing Hindi Translator Node ---")

    prompt = (
        "You are an expert content localizer for the Indian market. Take the following script "
        "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence-by-sentence "
        "or repeat information. Alternating comfortably between Hindi and English phrases just like "
        "an intellectual tech educator would speak naturally on a live stream. Keep the energy high! "
        "Return only the final Hinglish text.\n\n"
        f"Script:\n{state['script_text']}"
    )

    response = llm.invoke(prompt)

    return {
        "final_output": response.content.strip()
    }


# --------------------------------------------------
# 6. Create the LangGraph
# --------------------------------------------------

graph = StateGraph(PipelineState)


# Add nodes
graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)


# --------------------------------------------------
# 7. Connect the Nodes
# --------------------------------------------------

graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)


# --------------------------------------------------
# 8. Compile the Graph
# --------------------------------------------------

app = graph.compile()


# --------------------------------------------------
# 9. Run the Pipeline
# --------------------------------------------------

result = app.invoke(
    {
        "raw_input": (
            "AI agents are the future of technology. "
            "They can think, plan, and act on their own. "
            "LangGraph helps you build these agents with "
            "proper control, memory, and structured workflows."
        )
    }
)


# --------------------------------------------------
# 10. Display the Final Output
# --------------------------------------------------

print("\n========================================")
print("FINAL HINGLISH SCRIPT")
print("========================================\n")

print(result["final_output"])