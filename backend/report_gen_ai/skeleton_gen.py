from typing import List

from langchain_core.prompt_values import PromptValue
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

SYSTEM_PROMPT = (
    "Your task is to create a structured skeleton for a legal intake report based on a conversation between a client "
    "and an AI intake assistant. Your primary goal is to ensure organisational clarity while avoiding repetition of facts "
    "across different sections.\n\n"
    
    "PROCESS:\n\n"
    
    "1. REVIEW AND CATEGORIZE THE CONVERSATION:\n"
    "Analyze the conversation to identify key legal information. Categorize data into these primary themes:\n"
    "- Matter identification (type, parties, jurisdiction)\n"
    "- Factual timeline (what happened, when, where)\n"
    "- Legal status (existing orders, proceedings, representation)\n"
    "- Financial context (assets, liabilities, income, funding capacity)\n"
    "- Risk factors (safety, urgency, vulnerabilities)\n"
    "- Evidence and documentation\n"
    "- Client objectives and concerns\n\n"
    
    "2. IDENTIFY REPORT SECTIONS:\n"
    "Create distinct, non-overlapping sections. Standard legal intake report structure includes:\n"
    "- Matter Summary (brief overview only)\n"
    "- Parties Involved\n"
    "- Chronology of Events\n"
    "- Current Legal Position\n"
    "- Financial Overview\n"
    "- Risk Assessment\n"
    "- Available Evidence\n"
    "- Client Instructions (what they want)\n"
    "- Recommended Next Steps\n\n"
    
    "Tailor these sections to the specific matter type (family law, estate dispute, etc.).\n\n"
    
    "3. CREATE CLEAR HEADINGS:\n"
    "Develop descriptive headings and subheadings that reflect legal categories, not conversation flow. "
    "Headings should be specific (e.g., 'Property and Assets' not 'Financial Information').\n\n"
    
    "4. ALLOCATE FACTS TO SECTIONS:\n"
    "Place each fact under the most relevant section ONCE only. Guidelines:\n"
    "- Dates and events → Chronology of Events\n"
    "- Asset values → Financial Overview\n"
    "- Safety concerns → Risk Assessment\n"
    "- What client wants → Client Instructions\n"
    "If a fact relates to multiple sections, choose the PRIMARY section and use cross-references elsewhere.\n\n"
    
    "5. CRITICAL: AVOID REPETITION:\n"
    "Each fact appears in ONE section only. Mark facts as allocated. If a fact must be mentioned in multiple places, "
    "state it fully in one section and use brief cross-references elsewhere (e.g., 'See Financial Overview for asset details').\n\n"
    
    "6. MATTER SUMMARY:\n"
    "Write a 2-3 sentence overview stating: matter type, key issue, and urgency level. Do NOT include specific facts, "
    "dates, or amounts - those belong in detailed sections.\n\n"
    
    "7. RECOMMENDED NEXT STEPS:\n"
    "Based on urgency, risk, and legal status, outline immediate actions and follow-up items. Do NOT repeat facts from "
    "other sections - reference them (e.g., 'Given the DV allegations detailed in Risk Assessment...').\n\n"
    
    "8. LEGAL CLARITY:\n"
    "- Use plain language, not legal jargon\n"
    "- State facts objectively (what client said, not legal conclusions)\n"
    "- Distinguish between verified facts and client claims\n"
    "- Flag missing information as gaps\n\n"
    
    "9. CROSS-REFERENCING:\n"
    "When analysis requires facts from another section, use: 'As detailed in [Section Name]...' "
    "Do not repeat the actual facts.\n\n"
    
    "10. REVIEW FOR COMPLETENESS:\n"
    "Ensure:\n"
    "- All extracted facts are placed in appropriate sections\n"
    "- No fact appears verbatim in multiple sections\n"
    "- Logical flow from matter identification → facts → risk → recommendations\n"
    "- Gaps in information are clearly noted\n\n"
    
    "OUTPUT FORMAT:\n"
    "Provide a skeleton with:\n"
    "- Section headings and subheadings\n"
    "- Brief description of what belongs in each section\n"
    "- Bullet point placeholders showing which facts go where\n"
    "- Cross-reference notes where needed\n\n"
    
    "This skeleton will be used to generate the full legal intake report.")

class Section(BaseModel):
    heading: str = Field(...,
                         description="Heading of the section")
    sub_headings: List[str] = Field(...,
                                    description="Sub-headings in the section.")


class ReportSkeleton(BaseModel):
    skeleton: List[Section] = Field(...,
                                    description="Skeleton of the report consisting of all sections")
   

async def design_report_skeleton(serialized_conversation: str, llm: ChatOpenAI | None = None):
    output_parser = PydanticOutputParser(pydantic_object=ReportSkeleton)
    format_instructions = output_parser.get_format_instructions()

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(
                "{format_instructions}\n{user_prompt}")
        ],
        input_variables=["user_prompt"],
        partial_variables={"format_instructions": format_instructions}
    )
    prompt = prompt.format_prompt(user_prompt=serialized_conversation)

    output = await llm.ainvoke(prompt.to_messages())

    parsed_output = output_parser.parse(output.content)

    return parsed_output





   
