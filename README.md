# graphrag
 Knowledge graph-enhanced Retrieval-Augmented Generation (RAG) 


Knowledge Graph visuallization for Individual Income Tax returns for the year 2019 - https://github.com/aswinaus/graphrag/blob/main/Graph_RAG.ipynb : 
State Individual Income Tax has six Income ranges as per gov data. Each state is represented as a node which is related to other nodes such as No of tax returns, No of single tax returns and No of joint returns with the income range(Size of adjusted gross income) set as the relationship with visualization using networkx

 ![image](https://github.com/user-attachments/assets/3eb7ff32-a6a2-4f2c-9bb4-76de27426639)

 Extending the relation to zipcode and Tax returns through STATE as relationship with visualization using networkx

![image](https://github.com/user-attachments/assets/1013cdda-575a-4ee4-8da8-8b8d0c3e7aae)

Persisting the above Graph in Neo4J graph database for building question-answering system that uses Income Tax returns data stored as knowledge graph in Neo4j graph database

https://github.com/aswinaus/graphrag/blob/main/Neo4J%20representation%20of%20Income%20Tax%20returns%20data%20by%20State.png



