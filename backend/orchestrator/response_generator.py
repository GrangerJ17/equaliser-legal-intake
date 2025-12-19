from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .schemas import QuestionSchema, FieldCompletenessTracker
from langchain_core.output_parsers import StrOutputParser


class ResponseGenerator:
    """Generates responses based on mode"""
    
    def __init__(self, llm, system_prompt, rag_handler=None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.rag_handler = rag_handler
        
        self.chat_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),

            MessagesPlaceholder(variable_name="history"),

            ("system",
             
            "Please generate a natural empathic response that guides the you and the user to a better understanding of their situation"

            
            "Current user intent: {intent}\n"
            "User sentiment: {sentiment}\n"
            "Retrieved Facts: {context}"
            "Missing case facts:\n {missing}"
            ),

            ("human", "{input}")
        ])
    
    def listen(self, user_input: str, intent: str, history, completion_tracker: FieldCompletenessTracker, context = None) -> str:
        """Standard empathetic chat response"""
        
        chain = self.chat_template | self.llm
        response = chain.invoke({
                "history": history.messages,        # List[BaseMessage]
                "intent": intent,
                "context": context,
                "sentiment": completion_tracker.user_emotions,
                "input": user_input,
                "missing": completion_tracker.missing_critical_fields
            })
            
        
        return response.content
    
    def educate(self, user_input: str, intent: str, history, completion_tracker=None) -> str:
        """Provide factual legal information with RAG"""
        
        if not self.rag_handler:
            return self.listen(user_input, "seeking_information", history)
        
        # Retrieve relevant context
        context = self.rag_handler.retrieve(user_input)
        
        response = self.listen(user_input, intent, history, context=context, completion_tracker=completion_tracker)
        
        return response
    
    def guide(self, user_input: str, intent: str, completion_tracker: FieldCompletenessTracker, history) -> str:
        """Generate multiple choice questions"""
        
        parser = PydanticOutputParser(pydantic_object=QuestionSchema)
        
        template = PromptTemplate(
            template="""Based on user's intentions, create 4 questions prompting them to select what's most important.
            
            Format: {format}
            
            Chat history: {history}
            User Input: {input}
            User Intention: {intent}
            Missing fields: {missing}
            Unclear fields: {unclear}
            """,
            input_variables=["history", "input", "intent", "missing", "unclear"],
            partial_variables={"format": parser.get_format_instructions()}
        )
        
        chain = template | self.llm | parser
        
        result = chain.invoke({
            "history": history.messages,
            "input": user_input,
            "intent": intent,
            "missing": completion_tracker.missing_critical_fields,
            "unclear": completion_tracker.uncertain_fields
        })
        
        # Format as text
        questions_text = "To guide this properly, please tell me what feels most important:\n\n"
        for i, q in enumerate(result.questions, 1):
            questions_text += f"{i}. {q}\n"
        
        

        return questions_text
    