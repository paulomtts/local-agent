:warning: This is highly dependent on LLM model choice. At 01/03/2026, gpt-5-mini with 'low' reasoning effort was a good choice. :warning:

# Memory System Architecture

```mermaid
graph TB
    %% Inputs
    UserMsg[User Message] --> Hook
    AgentMsg[Agent Response / Tool Call] --> Hook

    Hook[after_append Hook] --> WriteWorking[Write working.md]
    Hook --> CheckTrivial{Non-trivial<br/>user message?}
    Hook --> CheckTokens{Tokens ><br/>threshold?}

    %% Extraction — user message path only
    CheckTrivial -->|Yes| SemanticExtract[Semantic Extraction<br/>LLM → atomic facts]
    CheckTrivial -->|Yes| EpisodicExtract[Episodic Extraction<br/>LLM → user intent]
    CheckTrivial -->|No| Skip[Skip extraction]

    %% Deterministic agent action logging
    AgentMsg -->|tool completions| EpisodicDirect[Deterministic Log<br/>no LLM]

    %% Compaction
    CheckTokens -->|Yes| Compact[Compact Memory<br/>LLM → summarize old items]
    Compact --> WriteWorking

    %% Storage
    SemanticExtract --> SemanticFile[".memory/semantic.md<br/>Timeless facts"]
    EpisodicExtract --> EpisodicFile[".memory/episodic.md<br/>Timestamped events"]
    EpisodicDirect --> EpisodicFile
    WriteWorking --> WorkingFile[".memory/working.md<br/>Current conversation"]

    %% Retrieval
    subgraph Retrieval["Memory Usage (During Agent Execution)"]
        Think[think tool<br/>decide what to do next]
        Respond[respond tool<br/>generate reply]
    end

    WorkingFile -->|recent context| Think
    EpisodicFile -->|recent events| Think
    WorkingFile -->|conversation pairs| Respond
    EpisodicFile -->|recent events| Respond
    SemanticFile -->|all facts| Respond

    %% Styling
    classDef decision fill:#ff9800,stroke:#e65100,stroke-width:2px,color:#fff
    classDef storage fill:#4caf50,stroke:#1b5e20,stroke-width:2px,color:#fff
    classDef retrieval fill:#00bcd4,stroke:#006064,stroke-width:2px,color:#fff

    class CheckTrivial,CheckTokens decision
    class WorkingFile,SemanticFile,EpisodicFile storage
    class Think,Respond retrieval
```

## System Overview

A single `after_append` hook (`app/memory/hooks.py`) fires on every context queue append. It writes the updated conversation to `working.md`, then decides whether to run extraction:

- **Non-trivial user message** → runs semantic + episodic extraction in parallel via LLM
- **Token threshold exceeded** → runs compaction (summarizes old items, keeps last 5)
- **Tool completions** → episodic events logged deterministically (no LLM)
- **Trivial messages** ("ok", "yes", "thanks") → skip extraction

### Memory Types

| Type | File | Purpose | Written by |
|------|------|---------|------------|
| **Working** | `working.md` | Active conversation (U/A/T/C items) | Every append |
| **Semantic** | `semantic.md` | Timeless, normalized facts | LLM on user message |
| **Episodic** | `episodic.md` | Timestamped events | LLM (user intent) + deterministic (agent actions) |

### Hybrid Episodic Extraction

User actions require LLM interpretation; agent actions don't — we already know what tools ran. This halves LLM calls for episodic logging.

```python
# Tools log their own events directly:
from app.memory import log_episodic_event
log_episodic_event(event="agent read 3 files", context="config.py, main.py, utils.py")
```

### Storage Format

```
# working.md
U: Hey, do you know Nina?
A: Yes, Nina is a Pomeranian born around 2011.

# semantic.md
- User has a dog named Nina
- Nina is a Pomeranian born around 2011
- Nina's fur is orange

# episodic.md
## 02-16 15:28
- user asked about Nina
- user provided Nina's fur color | orange
```
