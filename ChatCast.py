# ─────────────────────────────────────────────────────────────────────────────
# ChatCast Podcast Script Generator & Narrator
# ─────────────────────────────────────────────────────────────────────────────

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState
from murf import Murf
import os
from dotenv import load_dotenv

def load_api_keys():
    """Load API keys from .env into environment variables."""
    load_dotenv()
    return {
        "google": os.getenv("GOOGLE_API_KEY"),
        "murf": os.getenv("MURF_API_KEY")
    }

def init_llm(api_key: str, model: str = "gemini-1.5-flash-8b") -> ChatGoogleGenerativeAI:
    """Initialize the Google Gemini LLM."""
    return ChatGoogleGenerativeAI(model=model, api_key=api_key)

def parse_pdf(path: str) -> str:
    """
    Read all text from a PDF file.
    
    Args:
        path: Path to the PDF file.
    Returns:
        The concatenated text of all pages.
    """
    print("Extracting text from PDF...")
    reader = PdfReader(path)
    return "".join(page.extract_text() or "" for page in reader.pages)

class AgentState(MessagesState):
    """State holding the raw text, its summary, and the generated script."""
    text: str
    summary: str
    script: str

def summarize_sections(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """
    Split the PDF text into chunks, summarize each section, and return bullet points.
    
    Args:
        state: AgentState containing `state.text`.
        llm: Initialized ChatGoogleGenerativeAI instance.
    Returns:
        Updated AgentState with `state.summary`.
    """
    print("Summarizing PDF...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = splitter.create_documents([state["text"]])
    
    prompt_template = """
    You are a helpful assistant that summarizes research papers.
    Here are the document chunks:
    {text}
    
    Summarize each logical section in bullet points under its section name.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = (prompt | llm | StrOutputParser())
    
    # Invoke the chain on the list of docs to get a structured summary.
    summary = chain.invoke({"text": docs})
    return AgentState(text=state["text"], summary=summary, script="")

def style_transfer(state: AgentState, llm: ChatGoogleGenerativeAI) -> AgentState:
    """
    Turn an academic summary into a conversational Host/Expert podcast script.
    
    Args:
        state: AgentState containing `state.summary`.
        llm: Initialized ChatGoogleGenerativeAI instance.
    Returns:
        Updated AgentState with `state.script`.
    """
    print("Style-transfering summary...")
    prompt = f"""
    You are a podcast host in dialogue with an expert. Podcast: "ChatCast".
    Rewrite this academic summary into a conversational Host/Expert script.
    Expert name: Michael Scott.

    {state["summary"]}
    """
    script = llm.predict(prompt)
    return AgentState(text=state["text"], summary=state["summary"], script=script)

def build_workflow(llm: ChatGoogleGenerativeAI) -> StateGraph:
    """
    Construct and compile the state graph for summarization → style transfer.
    
    Returns:
        A compiled StateGraph ready for invocation.
    """
    graph = StateGraph(AgentState)
    graph.add_node("summarize", lambda s: summarize_sections(s, llm))
    graph.add_node("style_transfer", lambda s: style_transfer(s, llm))
    graph.set_entry_point("summarize")
    graph.add_edge("summarize", "style_transfer")
    return graph.compile()

def generate_audio(script: str, murf_key: str, output_path: str = "stream_output.wav"):
    """
    Stream text-to-speech audio for each speaker line in the script.
    
    Args:
        script: The full Host/Expert script with lines prefixed by "**Host:**" or "**Michael Scott:**".
        murf_key: API key for Murf.
        output_path: File to append the audio chunks to.
    """
    print("Generating audio...")
    client = Murf(api_key=murf_key)
    
    # Split script into lines and process each speaker turn
    for line in script.split("\n"):
        if line.startswith("**Host:**"):
            voice = "en-US-natalie"
            text = line.split("**Host:**", 1)[1].strip()
        elif line.startswith("**Michael Scott:**"):
            voice = "en-US-terrell"
            text = line.split("**Michael Scott:**", 1)[1].strip()
        elif line.startswith("**Michael:**"):
            voice = "en-US-terrell"
            text = line.split("**Michael:**", 1)[1].strip()
        else:
            continue
        
        # Stream and append audio
        stream = client.text_to_speech.stream(text=text, voice_id=voice)
        for chunk in stream:
            with open(output_path, "ab") as f:
                f.write(chunk)
        

def main(pdf_path: str):
    """
    Full end-to-end pipeline:
      1. Load keys & init LLM
      2. Parse PDF
      3. Summarize & style-transfer to script
      4. Generate podcast audio
    """
    keys = load_api_keys()
    llm = init_llm(keys["google"])
    
    # 1. Parse and summarize
    raw_text = parse_pdf(pdf_path)
    initial_state = AgentState(text=raw_text, summary="", script="")
    workflow = build_workflow(llm)
    final_state = workflow.invoke(initial_state)
    print(final_state["script"])
    # 2. Text-to-speech
    generate_audio(final_state["script"], keys["murf"])
    print("Podcast audio generated at 'stream_output.wav'")

if __name__ == "__main__":
    main("yolo.pdf")
