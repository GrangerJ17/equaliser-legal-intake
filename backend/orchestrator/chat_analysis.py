from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .schemas import MessageIntent, FieldCompletenessTracker, CaseFactsSchema
from typing import Tuple

class ConversationAnalyser:
    """Handles all analysis: intent, sentiment, completion"""
    
    def __init__(self, llm):
        self.llm = llm
        self.intent_parser = PydanticOutputParser(pydantic_object=MessageIntent)
        self.completion_parser = PydanticOutputParser(pydantic_object=FieldCompletenessTracker)
        self.state_parser = PydanticOutputParser(pydantic_object=CaseFactsSchema)
    def analyse_intent(self, history) -> Tuple[str, str]:
        """Determine user intent and suggested response mode"""
        
        template = PromptTemplate(
            template='''From the chat history, determine user's intent.
            
            Field explanations: 
            * listen = default mode, should be prefered unless otherwise
            * guide = Give questions if diverging intentions
            * educate = Use RAG to give factual answer
            
            Format: {format}
            CHAT HISTORY: {history}
            ''',
            input_variables=['history'],
            partial_variables={"format": self.intent_parser.get_format_instructions()}
        )
        
        chain = template | self.llm | self.intent_parser
        result = chain.invoke({"history": history})
        
        return result.primary_intent, result.suggested_response_style
    
    def analyse_completion(self, history, curr_case_facts):
        """Check if conversation is complete"""
        
        template = PromptTemplate(
            template='''From chat history, fill out this schema 
            
            Format: {format}
            Chat History: {history}
            ''',
            input_variables=['history'],
            partial_variables={"format": self.state_parser.get_format_instructions()}
        )
        
        chain = template | self.llm | self.state_parser
    
        case_facts = chain.invoke({"history": history })

        print(case_facts)

        for field, value in case_facts:
            setattr(curr_case_facts, field, value)
            
        
        template = PromptTemplate(
            template='''From gathered case facts, determine if the chat should end
            
            Format: {format}
            Chat Metrics: {data}
            ''',
            input_variables=['data'],
            partial_variables={"format": self.completion_parser.get_format_instructions()}
        )
        
        chain = template | self.llm | self.completion_parser
        completion = chain.invoke({"data": case_facts})
        
        print(completion)

        return completion, case_facts 
       
    def analyse_sentiment(self, text: str) -> dict:
        """Analyse emotional state"""
        # TODO: Implement when sentiment model ready
        return {"emotion": "neutral", "intensity": 0.5}