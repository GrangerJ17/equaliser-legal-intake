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
    
    reason_not_ready: Optional[str] = Field(
        default=None,
        description="Give reasons why the conversation cannot complete just yet"
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