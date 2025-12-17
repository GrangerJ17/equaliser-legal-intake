class RAGHandler:
    """Handles RAG operations"""
    
    def __init__(self, embedder):
        self.embedder = embedder
    
    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context for query"""
        
        results = self.embedder.retriever.invoke(query)
        
        context = "\n\n".join([
            chunk.page_content 
            for chunk in results[:top_k]
        ])
        
        return context