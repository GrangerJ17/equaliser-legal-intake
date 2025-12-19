from typing import Dict

# change this
from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate

from langchain_core.language_models.chat_models import BaseChatModel


SECTION_SYSTEM_PROMPT_TEMPLATE = (
    "Your task is to draft the {__SECTION_HEADING__} section of a legal intake report. "
    "You will be provided with:\n"
    "A structured skeleton defining what this section should contain\n"
    "A conversation between a client and an AI intake assistant\n"
    "The current draft of the intake report\n\n"

    "PURPOSE:\n"
    "This section must clearly and objectively capture ONLY the information that belongs in the "
    "{__SECTION_HEADING__} section, in line with standard legal intake reporting practices.\n\n"

    "SECTION SCOPE AND STRUCTURE:\n"
    "Use ONLY the following subheadings, in the order provided:\n"
    "{__SUB_SECTION_HEADINGS__}\n"
    "Do not introduce new headings or merge content across subheadings\n"
    "Each subheading should address a distinct legal or factual category\n\n"

    "FACT ALLOCATION RULES (CRITICAL):\n"
    "Each fact must appear in ONE section only across the entire report\n"
    "Do NOT repeat or paraphrase facts that already appear elsewhere in the report\n"
    "If relevant information is already captured in another section, use a brief cross-reference only "
    "(e.g., 'As detailed in Financial Overview...')\n"
    "Do NOT restate dates, amounts, allegations, or events already recorded\n\n"

    "CONTENT GUIDELINES:\n"
    "Use plain, professional language suitable for a legal intake file\n"
    "Respond in full sentences, to construct a thorough report section"
    "State information objectively as client-reported facts, not legal conclusions\n"
    "Clearly distinguish between confirmed information and client assertions\n"
    "Identify and flag missing or unclear information as gaps\n"
    "Avoid commentary, advocacy, or analysis beyond the purpose of this section\n\n"

    "USE OF THE CONVERSATION:\n"
    "Extract only the information relevant to this section’s scope\n"
    "Ignore conversational flow, repetition, or irrelevant details\n"
    "Do not infer facts that are not explicitly stated\n\n"

    "OUTPUT FORMAT (RAW TEXT ONLY):\n"
    "Use the section heading and subheadings exactly as provided\n"
    "Present information in clear paragraphs or bullet points where appropriate\n"
    "Include brief cross-reference notes where necessary\n"

    "CURRENT REPORT CONTEXT:\n"
    "This is the current draft of the intake report:\n"
    "{__SERIALIZED_REPORT__}\n\n"

    "ABSOLUTE RESTRICTION:\n"
    "You must NOT repeat any information already present in the current report. "
    "This includes facts, figures, dates, descriptions, and wording. "
    "If the information already exists, reference the relevant section instead.\n\n"

)

async def design_section(conversation: str, section: Dict, report: str, llm):

    SYSTEM_PROMPT = SECTION_SYSTEM_PROMPT_TEMPLATE.format_map({
            "__SECTION_HEADING__": section.heading,
            "__SUB_SECTION_HEADINGS__": "; ".join(section.heading),
            "__SERIALIZED_REPORT__": report
        })
    prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{conversation}")
        ])

    response = llm.invoke(prompt.format(conversation=conversation))

    print(response.content)

    return response.content