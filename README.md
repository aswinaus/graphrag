# graphrag
 Knowledge graph-enhanced Retrieval-Augmented Generation (RAG) 


Knowledge Graph visuallization for Individual Income Tax returns for the year 2019 - https://github.com/aswinaus/graphrag/blob/main/Graph_RAG.ipynb : 
State Individual Income Tax has six Income ranges as per gov data. Each state is represented as a node which is related to other nodes such as No of tax returns, No of single tax returns and No of joint returns with the income range(Size of adjusted gross income) set as the relationship with visualization using networkx


1. Define Use Case and Data
   •	Use Case:  Income Tax data
   •	Data: Individual Tax filing data by state and zip code. Other data can be : Historical claims, customer profiles, policy details, disaster impact data, geographical data, social networks, weather patterns.
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

Knowledge Graph through networkx
Code imagine each **STATE** as a circle (a node) and you want to connect it to another circle representing the number of tax returns filed in that state. This line of code draws that connection (an edge) between the 'STATE' node and the 'No of returns' node. It also labels the connection with 'Size of adjusted gross income' to indicate the relationship between them.

Scenario:If the current row in the dataset has 'STATE' as 'CA' and 'No of returns' as 1000, this line would create an edge in the graph connecting a node labeled 'CA' to a node labeled '1000', and the edge would be labeled with 'Size of adjusted gross income'.

Nodes: In a graph, nodes (sometimes called vertices) are the fundamental entities. In this code, nodes represent things like U.S. states, or categories like "Number of Returns" or "Size of adjusted gross income."

Edges: Edges are the connections between nodes. They represent relationships. In this code, an edge connects a state node to a tax return statistic node, and the edge is labeled with a "Size of adjusted gross income" attribute.

![image](https://github.com/user-attachments/assets/457961b5-18b6-4d42-8629-b61e2cbf6b7b)



 ![image](https://github.com/user-attachments/assets/3eb7ff32-a6a2-4f2c-9bb4-76de27426639)

 Extending the relation to zipcode and Tax returns through STATE as relationship with visualization using networkx

![image](https://github.com/user-attachments/assets/1013cdda-575a-4ee4-8da8-8b8d0c3e7aae)

Persisting the above Graph in Neo4J graph database for building question-answering system that uses Income Tax returns data stored as knowledge graph in Neo4j graph database

https://github.com/aswinaus/graphrag/blob/main/Neo4J%20representation%20of%20Income%20Tax%20returns%20data%20by%20State.png



