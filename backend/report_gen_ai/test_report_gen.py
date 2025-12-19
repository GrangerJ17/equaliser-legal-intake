from typing import List
import json

from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.prompt_values import PromptValue
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI
import sys
import os
from report import generate_report

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.prompts import EQUALISER_SYSTEM_PROMPT
import secrets
import asyncio
from skeleton_gen import design_report_skeleton



MOCK_CONVERSATION = [
    # ============ OPENING ============
    {
        "role": "assistant",
        "content": "Hello, I'm Equaliser. I'm here to help understand your legal situation. What brings you here today?"
    },
    {
        "role": "user",
        "content": "My husband moved out a few weeks ago and now he's saying the house is his and that I might have to leave. I'm really stressed and don't know what my rights are."
    },
    # TESTS:
    # - Sentiment: stress, anxiety (~0.7)
    # - Intent: asking_for_help
    # - Matter: property / family law
    # - Urgency: moderate
    
    # ============ EMPATHY ============
    {
        "role": "assistant",
        "content": "I'm sorry you're dealing with this. Disputes about the family home can be very stressful. I’ll ask you some questions so I can point you in the right direction."
    },
    {
        "role": "user",
        "content": "Thank you. I just feel overwhelmed."
    },

    # ============ RELATIONSHIP STATUS ============
    {
        "role": "assistant",
        "content": "Can you tell me a bit about your relationship status? Are you separated, and how long were you together?"
    },
    {
        "role": "user",
        "content": "We're separated but not legally divorced. We were married for 8 years before he moved out."
    },
    # TESTS:
    # - matter_type: family_law
    # - matter_subtype: separation, property settlement
    # - relationship_duration: 8 years
    
    # ============ TIMELINE ============
    {
        "role": "assistant",
        "content": "When did he move out of the home?"
    },
    {
        "role": "user",
        "content": "About three weeks ago."
    },
    # TESTS:
    # - incident_start_date: ~3 weeks ago
    # - separation_recent: True
    
    # ============ PROPERTY OWNERSHIP ============
    {
        "role": "assistant",
        "content": "Do you and your husband own the home together?"
    },
    {
        "role": "user",
        "content": "Yes, we bought it together five years ago. Both our names are on the title."
    },
    # TESTS:
    # - property_assets: joint ownership
    # - key_issue: family home
    
    # ============ PROPERTY VALUE ============
    {
        "role": "assistant",
        "content": "Do you know roughly what the property is worth and how much is left on the mortgage?"
    },
    {
        "role": "user",
        "content": "It's probably worth around $850,000 and we owe about $520,000 on the mortgage."
    },
    # TESTS:
    # - property_value: $850,000
    # - mortgage_balance: $520,000
    # - net_equity: ~$330,000
    
    # ============ CURRENT LIVING ARRANGEMENTS ============
    {
        "role": "assistant",
        "content": "Are you currently living in the home?"
    },
    {
        "role": "user",
        "content": "Yes, I’m still living there. He’s staying with a friend but keeps saying I won’t be able to afford the house and that I’ll have to sell."
    },
    # TESTS:
    # - occupation_status: client in home
    # - dispute: forced sale threat
    
    # ============ LEGAL ACTION ============
    {
        "role": "assistant",
        "content": "Have either of you started any legal process yet, like a property settlement or mediation?"
    },
    {
        "role": "user",
        "content": "No, nothing formal. He just keeps sending messages saying the house should be sold and that he wants his share."
    },
    # TESTS:
    # - current_legal_proceedings: False
    # - negotiation_attempts: informal only
    
    # ============ LOCATION ============
    {
        "role": "assistant",
        "content": "Where are you located?"
    },
    {
        "role": "user",
        "content": "Sydney, around Parramatta."
    },
    # TESTS:
    # - jurisdiction: NSW
    # - court: Federal Circuit and Family Court (implied)
    
    # ============ DESIRED OUTCOME ============
    {
        "role": "assistant",
        "content": "What would you ideally like to happen with the house?"
    },
    {
        "role": "user",
        "content": "I want to stay in the home if possible, or at least not be forced out before things are sorted properly."
    },
    # TESTS:
    # - desired_outcome: retain home / fair settlement
    # - deal_breakers: housing stability
    
    # ============ EMPLOYMENT & INCOME ============
    {
        "role": "assistant",
        "content": "Are you currently working, and do you know roughly what your husband earns?"
    },
    {
        "role": "user",
        "content": "I work part-time as a teacher’s aide earning about $35,000 a year. He’s a tradesman and earns about $85,000."
    },
    # TESTS:
    # - income_disparity: True
    # - relevance: borrowing capacity, settlement split
    
    # ============ OTHER ASSETS ============
    {
        "role": "assistant",
        "content": "Are there any other significant assets or debts I should be aware of?"
    },
    {
        "role": "user",
        "content": "We have about $12,000 in joint savings, superannuation for both of us, and a credit card with around $8,000 owing."
    },
    # TESTS:
    # - asset_pool: moderate
    # - shared_debts: True
    
    # ============ COST CONCERNS ============
    {
        "role": "assistant",
        "content": "Are you concerned about the cost of getting legal help?"
    },
    {
        "role": "user",
        "content": "Yes, very. I don’t have much savings and I can’t afford expensive legal fees."
    },
    # TESTS:
    # - cost_barrier: True
    # - funding_needed: likely
    
    # ============ REASSURANCE ============
    {
        "role": "assistant",
        "content": "That’s completely understandable. There are options like legal aid, fixed-fee advice, or payment plans. Getting advice early can actually save money in the long run."
    },
    {
        "role": "user",
        "content": "That makes me feel a bit better."
    },
    # TESTS:
    # - hesitation addressed
    # - engagement maintained
    
    # ============ WILLINGNESS TO NEGOTIATE ============
    {
        "role": "assistant",
        "content": "Would you be open to negotiating a property settlement, or do you think this may need to go to court?"
    },
    {
        "role": "user",
        "content": "I’d prefer to negotiate, but I want to understand my rights first. I don’t want to agree to something unfair."
    },
    # TESTS:
    # - willingness_to_negotiate: True
    # - intent: advice-seeking, not avoidance
    
    # ============ COMPLETION OFFER ============
    {
        "role": "assistant",
        "content": "Thank you for explaining everything. I have enough information to connect you with a family lawyer who specialises in property settlements. They can explain your rights regarding the home and help protect your position. Would you like me to proceed with that?"
    },
    {
        "role": "user",
        "content": "Yes, please. I’d really appreciate that."
    },
    # TESTS:
    # - ready_to_proceed: True
    
    # ============ FINAL HANDOFF ============
    {
        "role": "assistant",
        "content": "Great. Here’s what will happen next:\n\n1. A family lawyer experienced in property and separation matters will contact you\n2. They’ll explain your legal rights to the home and next steps\n3. We’ll discuss cost options that suit your situation\n\nYour reference number is EQ-3912. Is the best number to reach you your mobile?"
    },
    {
        "role": "user",
        "content": "Yes, that’s fine. Thank you for your help."
    },
    # TESTS:
    # - Complete: True
    # - Lead quality: high (clear issue, assets involved, ready to proceed)
]


llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

async def main():
   skeleton = await asyncio.create_task(design_report_skeleton(MOCK_CONVERSATION, llm))
   
   report = await asyncio.create_task(generate_report(MOCK_CONVERSATION, skeleton, llm))

asyncio.run(main())