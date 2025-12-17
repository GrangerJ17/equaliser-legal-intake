from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .schemas import QuestionSchema

class ResponseGenerator:
    """Generates responses based on mode"""
    
    def __init__(self, llm, system_prompt, rag_handler=None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.rag_handler = rag_handler
        
        self.chat_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
    
    def listen(self, user_input: str, intent: str, history) -> str:
        """Standard empathetic chat response"""
        
        chain = self.chat_template | self.llm
        
        response = chain.invoke({
            "history": history.messages,
            "input": f"""Respond empathetically with natural language, then ask a follow-up question.
            
            User Intent: {intent}
            User Input: {user_input}
            """
        })
        
        return response.content
    
    def educate(self, user_input: str, history) -> str:
        """Provide factual legal information with RAG"""
        
        if not self.rag_handler:
            return self.listen(user_input, "seeking_information", history)
        
        # Retrieve relevant context
        context = self.rag_handler.retrieve(user_input)
        
        response = self.llm.invoke(f"""
        Using this legal context, answer the user's question factually:
        
        Context: {context}
        User Question: {user_input}
        
        Respond conversationally but accurately. Cite sources where relevant.
        """)
        
        return response.content
    
    def guide(self, user_input: str, intent: str, completion_tracker, history) -> str:
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
    
    def act(self) -> str:
        """Completion/exit message"""
        return "Based on what you've shared, I have enough information to connect you with the right help. Would you like to proceed?"