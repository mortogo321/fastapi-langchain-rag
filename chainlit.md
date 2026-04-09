# RAG Knowledge Assistant

Welcome to the **Retrieval-Augmented Generation** chat interface.

This assistant answers questions based on documents that have been ingested into the knowledge base. It retrieves the most relevant passages from your documents and uses an LLM to generate accurate, context-grounded answers.

## How It Works

1. **Upload documents** via the API (`POST /api/ingest`) to build the knowledge base
2. **Ask questions** here in the chat about your uploaded documents
3. **Get cited answers** grounded in the actual document content

## Features

- Answers are strictly based on ingested document content
- Source references are provided with each answer
- Supports PDF and plain text documents
- Uses similarity search to find the most relevant context

## Tips

- Be specific in your questions for more accurate answers
- If the assistant says it does not have enough information, try uploading more relevant documents
- You can ingest multiple documents to build a comprehensive knowledge base
