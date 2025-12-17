from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .schemas import MessageIntent, FieldCompletenessTracker
from typing import Tuple

class ConversationAnalyser:
    """Handles all analysis: intent, sentiment, completion"""
    
    def __init__(self, llm):
        self.llm = llm
        self.intent_parser = PydanticOutputParser(pydantic_object=MessageIntent)
        self.completion_parser = PydanticOutputParser(pydantic_object=FieldCompletenessTracker)
    
    def analyse_intent(self, history) -> Tuple[str, str]:
        """Determine user intent and suggested response mode"""
        
        template = PromptTemplate(
            template='''From the chat history, determine user's intent.
            
            Field explanations: 
            * guide = Give questions if diverging intentions
            * listen = Standard chat mode
            * act = Determine next actions (exit or continue)
            * educate = Give legally factual answer
            
            Format: {format}
            CHAT HISTORY: {history}
            ''',
            input_variables=['history'],
            partial_variables={"format": self.intent_parser.get_format_instructions()}
        )
        
        chain = template | self.llm | self.intent_parser
        result = chain.invoke({"history": history})
        
        return result.primary_intent, result.suggested_response_style
    
    def analyse_completion(self, history) -> FieldCompletenessTracker:
        """Check if conversation is complete"""
        
        template = PromptTemplate(
            template='''From chat history, determine completion status.
            
            Format: {format}
            CHAT HISTORY: {history}
            ''',
            input_variables=['history'],
            partial_variables={"format": self.completion_parser.get_format_instructions()}
        )
        
        chain = template | self.llm | self.completion_parser
        result = chain.invoke({"history": history})
        
        return result
    
    def analyse_sentiment(self, text: str) -> dict:
        """Analyse emotional state"""
        # TODO: Implement when sentiment model ready
        return {"emotion": "neutral", "intensity": 0.5}