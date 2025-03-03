# graphrag
 Knowledge graph-enhanced Retrieval-Augmented Generation (RAG) 


Knowledge Graph visuallization for Individual Income Tax returns for the year 2019 - https://github.com/aswinaus/graphrag/blob/main/Graph_RAG.ipynb : 
State Individual Income Tax has six Income ranges as per gov data. Each state is represented as a node which is related to other nodes such as No of tax returns, No of single tax returns and No of joint returns with the income range(Size of adjusted gross income) set as the relationship with visualization using networkx


1. Define Use Case and Data
   •	Use Case:  Income Tax data
   •	Data: Historical claims, customer profiles, policy details, disaster impact data, geographical data, social networks, weather patterns.
3. Create and Populate Knowledge Graph
 •	Data Collection: Gather data from internal and external sources.
 •	Data Modeling: Define schema for entities and relationships.
 •	Data Ingestion: Load data into the knowledge graph.
4. Index and Embed Data
 •	Document Indexing: Index relevant documents.
 •	Embedding Creation: Generate embeddings for entities and relationships.
5. Set Up Retrieval Systems
 •	Document Retrieval: Implement system to retrieve documents from vector store.
 •	Graph Retrieval: Implement graph queries to extract relevant entities and relationships.
6. Develop Ranking and Filtering Algorithms
 •	Document Ranking: Rank and select top documents.
 •	Graph Ranking: Rank and filter graph data.
7. Integrate with Language Model
 •	Combine Data: Merge retrieved information from both sources.
 •	Response Generation: Use a language model to generate the final response.

Pending
7. Develop User Interface
•	Frontend: Create user-friendly interface.
•	Backend: Ensure seamless communication between components.
8. Testing and Validation
•	Test Scenarios: Validate accuracy and relevance. - Evals
•	User Feedback: Refine system based on feedback.
9. Deployment and Monitoring
•	Deployment: Deploy in production.
•	Monitoring: Continuously monitor and improve.


 ![image](https://github.com/user-attachments/assets/3eb7ff32-a6a2-4f2c-9bb4-76de27426639)

 Extending the relation to zipcode and Tax returns through STATE as relationship with visualization using networkx

![image](https://github.com/user-attachments/assets/1013cdda-575a-4ee4-8da8-8b8d0c3e7aae)

Persisting the above Graph in Neo4J graph database for building question-answering system that uses Income Tax returns data stored as knowledge graph in Neo4j graph database

https://github.com/aswinaus/graphrag/blob/main/Neo4J%20representation%20of%20Income%20Tax%20returns%20data%20by%20State.png



