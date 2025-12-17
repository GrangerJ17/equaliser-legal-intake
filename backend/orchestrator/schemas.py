from pydantic import BaseModel, Field
from typing import Optional, List, Literal

necessary_fields = ["matter_type","key_dates",
        "parties_involved",
        "desired_outcome", "income_sources",
        "estimated_monthly_income",
        "funding_concerns","risk_level",
        "risks_identified",
        "urgent_action_required",
        "urgency_score",
        "deadline_dates",
        "recommended_next_steps"
        ]

class FieldCompletenessTracker(BaseModel):
    """LLM fills this to track extraction quality"""
    
    # Core completeness metrics
    fields_filled_count: int = Field(
        default=0,
        description="How many required fields have meaningful values"
    )
    
    fields_total_count: int = Field(
        default=0,
        description="Total number of required fields for this phase"
    )
    
    completeness_ratio: float = Field(
        default=0.0,
        description="Ratio of filled to total fields (0.0 to 1.0)"
    )
    
    # Quality indicators
    confidence_level: str = Field(
        default="low",
        description="Overall confidence in extracted data: 'low', 'medium', or 'high'"
    )
    
    missing_critical_fields: List[str] = Field(
        default_factory=list,
        description=f"""List of fields that are empty but critical: {necessary_fields}"""
    )
    
    uncertain_fields: List[str] = Field(
        default_factory=list,
        description=f"Fields with values but low confidence: {necessary_fields}"
    )
    
    user_emotions: List[str] = Field(
        default=None,
        description="List of current user emotions and sentiments"
    )

class MessageIntent(BaseModel):
    primary_intent: Literal[
        "venting",
        "seeking_validation", 
        "asking_for_help",
        "exploring_options",
        "ready_to_proceed",
        "expressing_confusion"
    ]
    emotional_intensity: float 
    needs_reassurance: bool
    suggested_response_style: Literal["listen", "educate", "guide", "act"]

class QuestionSchema(BaseModel):
    questions: List[str]

    # # After each user message:
    # intent = llm.with_structured_output(MessageIntent).invoke([
    #     SystemMessage(content="Classify user intent and emotion"),
    #     HumanMessage(content=user_input)
    # ])

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import date

class CaseFactsSchema(BaseModel):
    """Comprehensive facts collected during legal intake"""
    
    # ============ MATTER IDENTIFICATION ============
    matter_type: Optional[Literal[
        "family_law",
        "estate_dispute", 
        "employment",
        "property_dispute",
        "litigation",
        "other"
    ]] = Field(default=None, description="Primary legal matter category")
    
    matter_subtype: Optional[str] = Field(
        default=None,
        description="Specific issue: divorce, will contest, unfair dismissal, etc."
    )
    
    brief_description: Optional[str] = Field(
        default=None,
        description="One-paragraph summary of the situation in user's own words"
    )
    
    # ============ PARTIES ============
    client_name: Optional[str] = Field(default=None, description="Client's full name")
    
    other_parties: List[dict] = Field(
        default_factory=list,
        description="Other involved parties: [{name, relationship, role}]"
    )
    
    children_involved: Optional[bool] = Field(
        default=None,
        description="Are children involved in this matter"
    )
    
    children_details: List[dict] = Field(
        default_factory=list,
        description="Children: [{name, age, current_custody}]"
    )
    
    # ============ TIMELINE ============
    incident_start_date: Optional[str] = Field(
        default=None,
        description="When did the issue begin (approximate if exact unknown)"
    )
    
    key_events: List[dict] = Field(
        default_factory=list,
        description="Chronological events: [{date, event_description}]"
    )
    
    upcoming_deadlines: List[dict] = Field(
        default_factory=list,
        description="Court dates, response deadlines: [{date, description, urgency}]"
    )
    
    statute_of_limitations_concern: Optional[bool] = Field(
        default=None,
        description="Is time running out to take legal action"
    )
    
    # ============ LEGAL STATUS ============
    current_legal_proceedings: Optional[bool] = Field(
        default=None,
        description="Are there active court proceedings"
    )
    
    court_orders_in_place: List[str] = Field(
        default_factory=list,
        description="Existing orders: protection orders, custody orders, etc."
    )
    
    previous_legal_action: Optional[str] = Field(
        default=None,
        description="Past legal steps taken on this matter"
    )
    
    represented_by_lawyer: Optional[bool] = Field(
        default=None,
        description="Does client currently have a lawyer"
    )
    
    # ============ FINANCIAL ============
    estimated_claim_value: Optional[str] = Field(
        default=None,
        description="Money/assets at stake (range acceptable)"
    )
    
    property_assets: List[dict] = Field(
        default_factory=list,
        description="Real estate, vehicles, businesses: [{type, value, ownership}]"
    )
    
    financial_accounts: List[dict] = Field(
        default_factory=list,
        description="Bank accounts, super, investments: [{type, approximate_value}]"
    )
    
    debts_liabilities: List[dict] = Field(
        default_factory=list,
        description="Mortgages, loans, credit cards: [{type, amount, shared}]"
    )
    
    client_employment_status: Optional[Literal[
        "employed_full_time",
        "employed_part_time",
        "self_employed",
        "unemployed",
        "retired",
        "student"
    ]] = Field(default=None)
    
    client_annual_income: Optional[str] = Field(
        default=None,
        description="Approximate annual income (range acceptable)"
    )
    
    ability_to_pay_legal_fees: Optional[Literal[
        "can_afford",
        "need_payment_plan",
        "need_funding",
        "legal_aid_eligible"
    ]] = Field(default=None)
    
    # ============ RISK FACTORS ============
    domestic_violence_present: Optional[bool] = Field(
        default=None,
        description="Is there DV, abuse, or safety concern"
    )
    
    immediate_safety_risk: Optional[bool] = Field(
        default=None,
        description="Is client or children in immediate danger"
    )
    
    risk_of_asset_dissipation: Optional[bool] = Field(
        default=None,
        description="Is other party hiding/selling assets"
    )
    
    mental_health_concerns: Optional[bool] = Field(
        default=None,
        description="Mental health issues affecting case (either party)"
    )
    
    substance_abuse_issues: Optional[bool] = Field(
        default=None,
        description="Drug/alcohol issues relevant to case"
    )
    
    risk_details: Optional[str] = Field(
        default=None,
        description="Additional context on identified risks"
    )
    
    # ============ EVIDENCE ============
    documentation_available: List[Literal[
        "contracts",
        "emails",
        "text_messages",
        "photos",
        "videos",
        "financial_records",
        "medical_records",
        "police_reports",
        "witness_statements",
        "other"
    ]] = Field(
        default_factory=list,
        description="Types of evidence client has"
    )
    
    witnesses_available: Optional[bool] = Field(
        default=None,
        description="Are there witnesses who can support the case"
    )
    
    evidence_quality: Optional[Literal["strong", "moderate", "weak", "none"]] = Field(
        default=None,
        description="Overall strength of available evidence"
    )
    
    # ============ CLIENT GOALS ============
    desired_outcome: Optional[str] = Field(
        default=None,
        description="What does client want to achieve"
    )
    
    primary_concerns: List[str] = Field(
        default_factory=list,
        description="Top 3 things client is most worried about"
    )
    
    deal_breakers: Optional[str] = Field(
        default=None,
        description="Non-negotiable items for client"
    )
    
    willingness_to_negotiate: Optional[Literal[
        "prefer_settlement",
        "willing_to_negotiate",
        "prepared_for_court",
        "court_only"
    ]] = Field(default=None)
    
    # ============ JURISDICTION ============
    state_territory: Optional[Literal[
        "NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"
    ]] = Field(default=None, description="Australian state/territory")
    
    matter_location: Optional[str] = Field(
        default=None,
        description="City/suburb where matter is based"
    )
    
    interstate_elements: Optional[bool] = Field(
        default=None,
        description="Does matter involve multiple states/territories"
    )
    
    # ============ URGENCY ============
    urgency_level: Optional[Literal[
        "crisis",      # Immediate danger, court tomorrow
        "urgent",      # Weeks
        "important",   # Months
        "routine"      # No time pressure
    ]] = Field(default=None)
    
    urgency_reason: Optional[str] = Field(
        default=None,
        description="Why is this urgent"
    )
    
    # ============ ADDITIONAL CONTEXT ============
    cultural_considerations: Optional[str] = Field(
        default=None,
        description="Cultural, religious, or linguistic factors to consider"
    )
    
    disability_accessibility_needs: Optional[str] = Field(
        default=None,
        description="Any accessibility requirements"
    )
    
    preferred_contact_method: Optional[Literal[
        "email", "phone", "text", "video_call"
    ]] = Field(default=None)
    
    additional_notes: Optional[str] = Field(
        default=None,
        description="Any other relevant information"
    )
    
    # ============ META ============
    facts_last_updated: Optional[str] = Field(
        default=None,
        description="Timestamp of last extraction"
    )
    
    confidence_score: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How confident are we in the collected facts (0-1)"
    )
    
    critical_gaps: List[str] = Field(
        default_factory=list,
        description="Essential information still missing"
    )