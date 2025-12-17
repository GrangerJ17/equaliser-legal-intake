from langchain_core.chat_history import InMemoryChatMessageHistory

class MemoryManager:
    """Manages all conversation memory"""
    
    def __init__(self, llm, condense_threshold=12):
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
        
        condensed_text = self.llm.invoke(f"""
        Condense this message history. Keep essential facts, remove redundancy:
        
        {self.short_term_memory.messages}
        """)
        
        new_history = InMemoryChatMessageHistory()
        new_history.add_ai_message(f"[Summary of previous conversation: {condensed_text.content}]")
        
        # Keep last 3 messages uncondesed
        for msg in self.short_term_memory.messages[-3:]:
            new_history.messages.append(msg)
        
        self.short_term_memory = new_history
        return new_history