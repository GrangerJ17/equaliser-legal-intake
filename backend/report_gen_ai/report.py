import os
import asyncio
from tqdm import tqdm
from pprint import pprint
from typing import List, Dict, Literal
from tenacity import retry, stop_after_attempt

from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.prompt_values import PromptValue
from pydantic import BaseModel, Field
from section_summarise import design_section

from langchain_openai import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI
import sys
import os


# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.prompts import EQUALISER_SYSTEM_PROMPT
import secrets
from skeleton_gen import design_report_skeleton

async def generate_section_content(conversation: str, section: Dict, llm: str):
    pass

# TODO: Format langchain messages to serialised dict

async def generate_report(conversation: List[dict], report_skeleton: Dict[str, str], llm: ChatOpenAI):

    report = []

    pprint(report_skeleton.skeleton)

    for section in tqdm(report_skeleton.skeleton, desc="Generating report..."):
        

        report_progress = '\n'.join(report)

        report_section =  await design_section(
            conversation=conversation, section=section, report=report_progress, llm=llm
        )

        report.append(report_section)

    return report


async def arun_generation(conversation: List[dict], title_dict: Dict[str, str], user_name: str, request_id: int, llm):
    pass
    