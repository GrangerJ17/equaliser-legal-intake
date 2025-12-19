from typing import List


from langchain_core.output_parsers import PydanticOutputParser

from langchain_core.prompt_values import PromptValue
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import ChatOpenAI
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.prompts import EQUALISER_SYSTEM_PROMPT
import secrets
import asyncio
from skeleton_gen import design_report_skeleton



MOCK_CONVERSATION = [
    # ============ OPENING - DISTRESS SIGNAL ============
    {
        "role": "assistant",
        "content": "Hello, I'm Equaliser. I'm here to help understand your legal situation. What brings you here today?"
    },
    {
        "role": "user",
        "content": "My husband left three weeks ago and took our kids. I don't know where they are and I'm terrified."
    },
    # TESTS: 
    # - Sentiment: fear (high intensity ~0.9)
    # - Intent: asking_for_help
    # - Risk: immediate_safety_risk, children_involved
    # - Urgency: crisis
    
    # ============ EMPATHY + SAFETY CHECK ============
    {
        "role": "assistant",
        "content": "I'm so sorry you're going through this. That sounds incredibly frightening. First, are you and the children safe right now? Is there any immediate danger?"
    },
    {
        "role": "user",
        "content": "I'm safe, but I don't know about the kids. He has a history of anger issues but never physically hurt them. I'm just so worried."
    },

    # TESTS:
    # - Risk: domestic_violence_present (emotional), mental_health_concerns
    # - Sentiment: fear, sadness (0.8)
    # - Evidence: witness to behavior patterns
    
    # ============ MATTER TYPE IDENTIFICATION ============
    {
        "role": "assistant",
        "content": "I understand. Let me help you get connected with someone who can assist. Can you tell me - are you and your husband separated, or is this the first time he's left?"
    },
    {
        "role": "user",
        "content": "We've been having problems for months but this is the first time he's actually left. We're not legally separated or anything. We've been married 8 years."
    },

    # TESTS:
    # - matter_type: family_law
    # - matter_subtype: separation, custody
    # - incident_start_date: ~3 weeks ago
    # - relationship_duration: 8 years
    
    # ============ CHILDREN DETAILS ============
    {
        "role": "assistant",
        "content": "Thank you for sharing that. Can you tell me about your children - how many and what ages?"
    },
    {
        "role": "user",
        "content": "Two daughters - Emma is 6 and Sophie is 4. They should be in school but I don't even know if he's taking them."
    },
    # TESTS:
    # - children_involved: True
    # - children_details: [{name: Emma, age: 6}, {name: Sophie, age: 4}]
    # - urgency: education disruption
    
    # ============ TIMELINE ============
    {
        "role": "assistant",
        "content": "I can understand why you're so worried. When exactly did he leave with the girls?"
    },
    {
        "role": "user",
        "content": "November 28th. He picked them up from school and never brought them back. I've tried calling but he won't answer."
    },
    # TESTS:
    # - key_events: [{date: "2024-11-28", event: "father took children from school"}]
    # - communication_breakdown: no contact
    # - upcoming_deadlines: potential for urgent parenting order
    
    # ============ LEGAL STATUS CHECK ============
    {
        "role": "assistant",
        "content": "Have you been to the police or court about this? Are there any existing court orders about custody or parenting arrangements?"
    },
    {
        "role": "user",
        "content": "No, nothing. I called the police but they said it's a 'civil matter' because he's their father. There's no court orders or anything like that."
    },
    # TESTS:
    # - current_legal_proceedings: False
    # - court_orders_in_place: []
    # - previous_legal_action: police report filed (no action)
    # - represented_by_lawyer: False
    
    # ============ LOCATION ============
    {
        "role": "assistant",
        "content": "Where are you located? This helps me connect you with the right local support."
    },
    {
        "role": "user",
        "content": "We're in Sydney, Parramatta area. That's where the kids' school is too."
    },
    # TESTS:
    # - state_territory: NSW
    # - matter_location: Parramatta, Sydney
    # - interstate_elements: False (so far)
    
    # ============ DESIRED OUTCOME ============
    {
        "role": "assistant",
        "content": "What would you most want to happen right now? What's your main priority?"
    },
    {
        "role": "user",
        "content": "I just want my girls back. I want them home where they're safe. I don't even care about the house or money, I just want to know they're okay and that I can see them."
    },
    # TESTS:
    # - desired_outcome: primary custody of children
    # - primary_concerns: child safety, wellbeing
    # - deal_breakers: must have access to children
    # - Intent: ready_to_proceed (high buying_readiness)
    # - Sentiment: determined (0.7), fear (0.6)
    
    # ============ FINANCIAL SITUATION - ASSETS ============
    {
        "role": "assistant",
        "content": "I understand. To find the right help, I need to understand your situation a bit better. Do you and your husband own your home?"
    },
    {
        "role": "user",
        "content": "Yes, we bought it 5 years ago. It's worth about $850,000 now but we still owe $520,000 on the mortgage. Both our names are on it."
    },
    # TESTS:
    # - property_assets: [{type: "family_home", value: "$850,000", ownership: "joint", debt: "$520,000"}]
    # - estimated_claim_value: Net equity ~$330,000
    # - debts_liabilities: [{type: "mortgage", amount: "$520,000", shared: True}]
    
    # ============ FINANCIAL - EMPLOYMENT & INCOME ============
    {
        "role": "assistant",
        "content": "Are you currently working?"
    },
    {
        "role": "user",
        "content": "I work part-time as a teacher's aide, about $35,000 a year. My husband is a tradesman, he makes about $85,000. But I don't know if he's still working or what's happening with money."
    },
    # TESTS:
    # - client_employment_status: employed_part_time
    # - client_annual_income: "$35,000"
    # - other_party_income: "$85,000"
    # - financial_accounts: unclear (red flag - no access)
    # - risk_of_asset_dissipation: possible
    
    # ============ OTHER ASSETS ============
    {
        "role": "assistant",
        "content": "Do you have any savings, superannuation, or other assets like cars?"
    },
    {
        "role": "user",
        "content": "We have about $12,000 in a joint savings account, but I don't know if he's touched it. He has his work ute that's paid off, probably worth $30,000. We each have our super - mine's about $45,000, his is probably $120,000 or so. That's about it."
    },
    # TESTS:
    # - financial_accounts: [
    #     {type: "joint_savings", value: "$12,000", access: "uncertain"},
    #     {type: "superannuation_client", value: "$45,000"},
    #     {type: "superannuation_spouse", value: "$120,000"}
    #   ]
    # - property_assets: [{type: "vehicle", value: "$30,000", ownership: "husband"}]
    # - Total asset pool: ~$550k equity + assets - debts
    
    # ============ OTHER DEBTS ============
    {
        "role": "assistant",
        "content": "Do you have any other debts - credit cards, car loans, anything like that?"
    },
    {
        "role": "user",
        "content": "Just a credit card with about $8,000 on it. Both our names but I think he's been using it. I'm worried about that too."
    },
    # TESTS:
    # - debts_liabilities: [{type: "credit_card", amount: "$8,000", shared: True}]
    # - risk_of_asset_dissipation: increasing concern
    
    # ============ ABILITY TO PAY ============
    {
        "role": "assistant",
        "content": "I need to ask about legal costs. Are you able to afford lawyer fees, or would you need help with funding?"
    },
    {
        "role": "user",
        "content": "I don't have much money. Maybe a couple thousand. I definitely can't afford big legal bills. Do I qualify for legal aid or something?"
    },
    # TESTS:
    # - ability_to_pay_legal_fees: need_funding
    # - Legal aid eligibility: likely (income <$45k, DV elements, children)
    # - Intent: still engaged but cost_concern barrier detected
    
    # ============ HESITATION - BARRIER ADDRESSED ============
    {
        "role": "assistant",
        "content": "That's a very common concern. There are funding options including legal aid, and payment plans. Cost shouldn't stop you from getting help, especially with children involved. Let's focus on your situation first, and we can work out funding that fits your circumstances."
    },
    {
        "role": "user",
        "content": "Okay, that's a relief. I was worried I couldn't afford to do anything."
    },
    # TESTS:
    # - Hesitation detected → barrier addressed → engagement restored
    # - Intent: back to ready_to_proceed
    # - Sentiment: relief (0.4), hope emerging
    
    # ============ RISK ASSESSMENT - DV DETAILS ============
    {
        "role": "assistant",
        "content": "You mentioned your husband has anger issues. Has there been any violence or threats in the relationship?"
    },
    {
        "role": "user",
        "content": "He's never hit me, but he's punched walls, thrown things. He screams at me in front of the kids. A few weeks before he left he said he'd 'make sure I never see the girls again' during an argument. That's why I'm so scared now."
    },
    # TESTS:
    # - domestic_violence_present: True (emotional, intimidation)
    # - immediate_safety_risk: moderate (threats made)
    # - risk_details: property damage, verbal abuse, threats re: children
    # - documentation_available: potential evidence (property damage photos?)
    # - Impact on urgency_level: crisis
    
    # ============ EVIDENCE CHECK ============
    {
        "role": "assistant",
        "content": "Do you have any evidence of this - text messages, photos, anyone who witnessed these incidents?"
    },
    {
        "role": "user",
        "content": "I have text messages where he's been really aggressive. And my sister has seen him yell at me before. I took a photo of the hole he punched in the wall last month."
    },
    # TESTS:
    # - documentation_available: ["text_messages", "photos"]
    # - witnesses_available: True (sister)
    # - evidence_quality: moderate
    
    # ============ SUBSTANCE ABUSE CHECK ============
    {
        "role": "assistant",
        "content": "Is there any drug or alcohol use that's relevant to the situation?"
    },
    {
        "role": "user",
        "content": "He drinks a lot, especially on weekends. Not every day but when he does he can get really angry. I don't think he does drugs but I'm not 100% sure."
    },
    # TESTS:
    # - substance_abuse_issues: True (alcohol)
    # - Impact on case strength: relevant to parenting capacity
    # - Evidence: potentially observable, witness testimony
    
    # ============ MENTAL HEALTH ============
    {
        "role": "assistant",
        "content": "How are you coping with all this? Are you getting any support?"
    },
    {
        "role": "user",
        "content": "I'm barely sleeping. I've been to my GP and she gave me something for anxiety. My sister has been helping but I'm struggling. I just need my kids back."
    },
    # TESTS:
    # - mental_health_concerns: True (client - anxiety from trauma)
    # - Support network: sister
    # - Sentiment: distress (0.8), exhaustion
    # - Medical documentation: GP records (evidence of impact)
    
    # ============ WILLINGNESS TO NEGOTIATE ============
    {
        "role": "assistant",
        "content": "If you could talk to your husband safely, would you be willing to work out an agreement, or do you think this needs to go to court?"
    },
    {
        "role": "user",
        "content": "I'd try to work something out if he'd actually talk to me, but he won't answer my calls. If that's what it takes to get my girls back, I'll go to court. I'll do whatever I have to."
    },
    # TESTS:
    # - willingness_to_negotiate: willing_to_negotiate (preferred) but prepared_for_court
    # - Communication status: broken down
    # - Intent: ready_to_proceed (very high buying_readiness ~0.9)
    # - Sentiment: determined (0.8)
    
    # ============ CULTURAL/ACCESSIBILITY ============
    {
        "role": "assistant",
        "content": "Is there anything else I should know - any cultural considerations, language needs, or accessibility requirements?"
    },
    {
        "role": "user",
        "content": "No, we're both Australian. English is fine. I just need help as soon as possible."
    },
    # TESTS:
    # - cultural_considerations: None
    # - disability_accessibility_needs: None
    # - preferred_contact_method: phone (implied urgency)
    
    # ============ COMPLETION OFFER ============
    {
        "role": "assistant",
        "content": "Thank you for sharing all of this with me. I know that wasn't easy. I have enough information to connect you with a family lawyer who specializes in urgent parenting matters and has experience with domestic violence cases. They can help you get an urgent court application to see your daughters. Would you like me to proceed with that?"
    },
    {
        "role": "user",
        "content": "Yes, please. How soon can someone help me?"
    },
    # TESTS:
    # - Completion check: ready_to_exit = True
    # - User confirmation: explicit "yes"
    # - Intent: ready_to_hire
    # - Engagement: high throughout
    
    # ============ FINAL HANDOFF ============
    {
        "role": "assistant",
        "content": "I'm connecting you with a specialist now. Based on your situation, this is what will happen next:\n\n1. A family lawyer experienced in urgent parenting matters will contact you within 2 hours\n2. They'll help you file for urgent court orders to locate the children and establish interim custody\n3. We'll connect you with legal aid funding given your circumstances\n4. Given the DV elements, they'll also help with safety planning\n\nYour reference number is: EQ-2847. Someone will call you shortly. Is the best number to reach you on [your number]?"
    },
    {
        "role": "user",
        "content": "Yes, that's my mobile. Thank you so much."
    },
    # TESTS:
    # - Complete: True
    # - Lead quality score: 95/100 (high urgency, clear need, strong case)
    # - Handoff successful with clear next steps
]

llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

async def main():
   pass 

asyncio.run(main())