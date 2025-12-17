from langchain_core.chat_history import InMemoryChatMessageHistory

class MemoryManager:
    """Manages all conversation memory"""
    
    def __init__(self, llm, condense_threshold=3):
        self.llm = llm
        self.condense_threshold = condense_threshold
        
        self.total_history = InMemoryChatMessageHistory()
        self.short_term_memory = InMemoryChatMessageHistory()
        self.user_only_history = InMemoryChatMessageHistory()
    
    def add_user_message(self, message: str):
        """Add user message to all relevant histories"""
        self.total_history.add_user_message(message)
        self.short_term_memory.add_user_message(message)
        self.user_only_history.add_user_message(message)
    
    def add_ai_message(self, message: str):
        """Add AI message to relevant histories"""
        self.total_history.add_ai_message(message)
        self.short_term_memory.add_ai_message(message)
    
    def get_short_term_history(self):
        """Get short-term history, condensing if needed"""
        
        if len(self.short_term_memory.messages) > self.condense_threshold:
            condensed = self._condense_history()
            return condensed
        
        return self.short_term_memory
    
    def _condense_history(self):
        """Condense history to reduce tokens"""

        essential_fields = [
                # Matter Identification
                "matter_type",
                "matter_subtype",
                "brief_description",
                
                # Parties
                "client_name",
                "other_parties",
                "children_involved",
                "children_details",
                
                # Timeline
                "incident_start_date",
                "key_events",
                "upcoming_deadlines",
                "statute_of_limitations_concern",
                
                # Legal Status
                "current_legal_proceedings",
                "court_orders_in_place",
                "previous_legal_action",
                "represented_by_lawyer",
                
                # Financial
                "estimated_claim_value",
                "property_assets",
                "financial_accounts",
                "debts_liabilities",
                "client_employment_status",
                "client_annual_income",
                "ability_to_pay_legal_fees",
                
                # Risk Factors
                "domestic_violence_present",
                "immediate_safety_risk",
                "risk_of_asset_dissipation",
                "mental_health_concerns",
                "substance_abuse_issues",
                "risk_details",
                
                # Evidence
                "documentation_available",
                "witnesses_available",
                "evidence_quality",
                
                # Client Goals
                "desired_outcome",
                "primary_concerns",
                "deal_breakers",
                "willingness_to_negotiate",
                
                # Jurisdiction
                "state_territory",
                "matter_location",
                "interstate_elements",
                
                # Urgency
                "urgency_level",
                "urgency_reason",
                
                # Additional Context
                "cultural_considerations",
                "disability_accessibility_needs",
                "preferred_contact_method",
                "additional_notes",
                
                # Meta
                "facts_last_updated",
                "confidence_score",
                "critical_gaps"
            ]
        
        condensed_text = self.llm.invoke(f"""
        Condense this message history. 
        Keep all facts relating to the following {essential_fields}
        
        Message History: {self.short_term_memory.messages}
        """)
        
        new_history = InMemoryChatMessageHistory()
        new_history.add_ai_message(f"[Summary of previous conversation: {condensed_text.content}]")
        
        # Keep last 3 messages uncondesed
        for msg in self.short_term_memory.messages[-3:]:
            new_history.messages.append(msg)
        
        self.short_term_memory = new_history

        print(new_history)

        return new_history