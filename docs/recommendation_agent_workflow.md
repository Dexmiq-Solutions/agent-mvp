flowchart TD
    Start([User Message]) --> P1a

    subgraph Legend ["Legend"]
        direction LR
        K1["Agent"]:::agent
        K2["Sub-Agent"]:::sub
        K3["Profile"]:::profile
        K4["Workflow Step"]:::step
    end

    subgraph Phase1 ["Phase 1: Understand the Request"]
        P1a["Load Context<br/><i>Conversation, BRD, decision history, user access</i>"]:::step
        P1a --> P1b["Resolve References<br/><i>Which section or point the user means</i>"]:::step
        P1b --> P1c["Intent Router Agent<br/><i>Selects the Profile, flags high-stakes decisions, lists missing facts</i>"]:::agent
        P1c --> P1d{"Missing facts or<br/>unclear request?"}
    end
    P1d -- "Yes" --> Ask(["Ask the User<br/><i>1-3 targeted questions. The answer resumes the same conversation</i>"])
    P1d -- "No" --> P2a

    subgraph Phase2 ["Phase 2: Apply the Profile"]
        P2a{"Selected Profile"}
        P2a -- "Explain" --> PX["Explain Profile<br/><i>Needs: what it says, where it came from, what depends on it<br/>Compares options: No</i>"]:::profile
        P2a -- "Recommend" --> PR["Recommend Profile<br/><i>Needs: goals, constraints, industry standards<br/>Compares options: Yes</i>"]:::profile
        P2a -- "Conflict" --> PC["Conflict Profile<br/><i>Needs: both requirements, their sources, standards<br/>Diagnose first, then compare resolutions</i>"]:::profile
        P2a -- "Communication" --> PM["Communication Profile<br/><i>Needs: cause, impact, new date, mitigation, audience<br/>Compares approaches</i>"]:::profile
    end
    PX --> G1
    PR --> G1
    PC --> G1
    PM --> G1

    subgraph Phase3 ["Phase 3: Gather Evidence"]
        N1["Lead Agent Rules<br/><i>Stay neutral. Hold position without new evidence.<br/>Never invent facts. Never edit the BRD directly</i>"]:::note
        N1 -.-> G1
        G1["Recommendation Lead Agent<br/><i>Gathers what the profile needs from BRD sections,<br/>project knowledge, industry standards, decision history</i>"]:::agent
        G1 --> G2{"Does the profile need<br/>an option comparison?"}
    end
    G2 -- "No: Explain, or Conflict with no real conflict" --> W1
    G2 -- "Yes" --> O1

    subgraph Phase4 ["Phase 4: Compare Options"]
        O1["Option Explorer Sub-Agent<br/><i>Proposes 2-3 independent options, including 'do nothing'.<br/>Never sees the user's preferred answer</i>"]:::sub
        O1 --> O2["Option Scoring and Ranking<br/><i>Scores each option against the profile's criteria, ranks them,<br/>checks the winner holds up</i>"]:::step
        O2 --> O3{"High-stakes<br/>decision?"}
        O3 -- "Yes" --> O4["Critic Sub-Agent<br/><i>Stress-tests the top option and looks for a missing option</i>"]:::sub
    end
    O3 -- "No" --> W1
    O4 --> W1

    subgraph Phase5 ["Phase 5: Write and Verify"]
        W1["Recommendation Lead Agent<br/><i>Writes the answer in the profile's format:<br/>answer first, options, pick, assumptions, sources, next actions</i>"]:::agent
        W1 --> V1["Verification Agent<br/><i>Checks every claim against the evidence that was gathered</i>"]:::agent
        V1 --> V2["Evidence Check<br/><i>Cited sources were really retrieved and the user may see them</i>"]:::step
        V2 --> V3{"Passes?"}
        V3 -- "No, first time" --> W1
        V3 -- "No, still failing" --> V4["Add Caveats<br/><i>Remove unsupported claims, lower confidence</i>"]:::step
    end
    V3 -- "Yes" --> R1
    V4 --> R1

    subgraph Phase6 ["Phase 6: Respond and Save"]
        R1["Render Answer<br/><i>Answer first, options table, confidence, assumptions, labeled sources</i>"]:::step
        R1 --> R2{"BRD change<br/>suggested?"}
        R2 -- "Yes" --> R3["Propose BRD Edits<br/><i>Shown as a diff for approval. Applied only by the BRD workflow</i>"]:::step
        R2 -- "No" --> R4
        R3 --> R4["Save Session<br/><i>Conversation state, decision record, trace and user feedback</i>"]:::step
    end
    R4 --> Done([Response to the User])
    Ask --> Done

    classDef agent fill:#e3f2fd,stroke:#1976d2,color:#000
    classDef sub fill:#f3e5f5,stroke:#8e24aa,color:#000
    classDef profile fill:#fff8e1,stroke:#f9a825,color:#000
    classDef step fill:#eceff1,stroke:#607d8b,color:#000
    classDef note fill:#ffffff,stroke:#9e9e9e,stroke-dasharray: 4 3,color:#000