from .chat_analysis import ConversationAnalyser
from .memory_manager import MemoryManager
from .response_generator import ResponseGenerator
from .rag_handler import RAGHandler
from .schemas import FieldCompletenessTracker
from transformers import pipeline

class ChatOrchestrator:
    """Main orchestrator - coordinates components"""
    
    def __init__(self, llm, assistant_llm, embedder, system_prompt):
        # Core components
        self.analyser = ConversationAnalyser(llm)
        self.memory = MemoryManager(llm)
        self.rag = RAGHandler(embedder)
        self.responder = ResponseGenerator(llm, system_prompt, self.rag)
        
        # State
        self.completion_tracker = FieldCompletenessTracker()
        self.complete = False
        self.message_count = 0
        self.message_limit = 50
    
    def orchestrate(self, user_input: str) -> str:
        """Main entry point - process user input and return response"""
        
        if self.complete:
            return "Thank you for providing this information. A specialist will be in touch."
        
        if not user_input:
            return None
        
        # 1. Save to memory
        self.memory.add_user_message(user_input)
        self.message_count += 1
        
        # 2. Analyse intent and sentiment
        intent, mode = self.analyser.analyse_intent(
            self.memory.user_only_history.messages
        )
        
        # 3. Generate response based on mode
        response = self._generate_response(user_input, intent, mode)
        
        # 4. Save AI response
        if response:
            self.memory.add_ai_message(response)
        
        # 5. Check completion
        self._check_completion()
        
        return response
    
    def _generate_response(self, user_input: str, intent: str, mode: str) -> str:
        """Route to appropriate response generator"""
        
        history = self.memory.get_short_term_history()
        
        if mode == "listen":
            return self.responder.listen(user_input, intent, history)
        
        elif mode == "educate":
            return self.responder.educate(user_input, history)
        
        elif mode == "guide":
            return self.responder.guide(
                user_input, 
                intent, 
                self.completion_tracker, 
                history
            )
        
        elif mode == "act":
            return self.responder.act()
        
        else:
            return self.responder.listen(user_input, intent, history)
    
    def _check_completion(self):
        """Check if conversation should end"""
        
        self.completion_tracker = self.analyser.analyse_completion(
            self.memory.short_term_memory.messages
        )
        
        # Exit conditions
        if len(self.completion_tracker.missing_critical_fields) == 0 and len(self.completion_tracker.missing_critical_fields) <= 2:
            classifier = pipeline("zero-shot-classification",
                      model="valhalla/distilbart-mnli-12-1")  # smaller than bart-large

            ai_message = """I believe I have enough information to draft a report of your situation.
            Would you like to continue to discuss your case or move on to finalising your report?"""

            print(ai_message)
            user_input = input("User Input: ")

            labels = ["false", "true"]
            result = classifier(user_input, candidate_labels=labels)

            print(result)

            if result == "true":
                self.complete = True
            else:
                self.complete = False 


        
        if self.message_count >= self.message_limit:
            self.complete = True